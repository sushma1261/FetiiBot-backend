import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import cors from "cors";
import dotenv from "dotenv";
import express, { Request, Response } from "express";
import { ConversationChain } from "langchain/chains";
import { Document } from "langchain/document";
import { BufferMemory, ChatMessageHistory } from "langchain/memory";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import multer from "multer";
import XLSX from "xlsx";
import { Rider, Trip, UserAge } from "./interfaces/interface";

dotenv.config();

const PORT = process.env.PORT ? parseInt(process.env.PORT) : 4000;

const app = express();
app.use(cors());
app.use(express.json());

const upload = multer({ storage: multer.memoryStorage() });

function readSheetCaseInsensitive(workbook: XLSX.WorkBook, name: string) {
  const lname = name.toLowerCase();
  for (const sname of workbook.SheetNames) {
    if (sname.toLowerCase() === lname)
      return XLSX.utils.sheet_to_json(workbook.Sheets[sname]);
  }
  return null;
}

let vectorStore: MemoryVectorStore | null = null;
let enrichedData: any[] = [];

/** Utility: standardize keys */
const standardizeKeys = (arr: any[]) =>
  arr.map((row) => {
    const newRow: any = {};
    Object.keys(row).forEach((k) => {
      newRow[k.trim().replace(/\s+/g, "_")] = row[k];
    });
    return newRow;
  });

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY!,
  model: "text-embedding-004",
});

/** Utility: merge sheets */
function mergeSheets(trips: any[], checkedIn: any[], demographics: any[]) {
  const tripsData = standardizeKeys(trips);
  const checkedInData = standardizeKeys(checkedIn);
  const demographicsData = standardizeKeys(demographics);

  const tripsWithCheckin = tripsData.map((trip) => {
    const checkin = checkedInData.find((c) => c.Trip_ID === trip.Trip_ID);
    return { ...trip, checkedInUserID: checkin ? checkin.User_ID : null };
  });

  return tripsWithCheckin.map((trip) => {
    const userDemo = demographicsData.find(
      (d) => d.User_ID === trip.checkedInUserID
    );
    return { ...trip, Age: userDemo ? userDemo.Age : null };
  });
}

// /** Utility: convert JS array to CSV stream */
// function convertToCSVStream(data: any[]) {
//   const csvString = Papa.unparse(data);
//   return Readable.from([csvString]);
// }

/** Utility: create vector store */
async function createVectorStore(data: any[]) {
  const docs = data.map((row) => {
    // Include TripDateISO explicitly in text for AI reasoning
    const rowWithDate = { ...row };
    if (row.TripDateISO) {
      rowWithDate.TripDateISO = row.TripDateISO;
    }

    const text = Object.entries(rowWithDate)
      .filter(([_, v]) => v !== undefined && v !== null)
      .map(([k, v]) => `${k}: ${v}`) // include key in text
      .join("; ");

    return new Document({ pageContent: text, metadata: row });
  });

  return MemoryVectorStore.fromDocuments(docs, embeddings);
}
function excelDateToJSDate(serial: number): Date {
  return new Date((serial - 25569) * 86400 * 1000);
}

app.post(
  "/upload",
  upload.single("file"),
  async (req: Request, res: Response) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
      }
      // Parse workbook
      const workbook = XLSX.read(req.file.buffer, { type: "buffer" });
      const trips = readSheetCaseInsensitive(workbook, "Trip Data") as Trip[];
      const riders = readSheetCaseInsensitive(
        workbook,
        "Checked in User ID's"
      ) as Rider[];

      const userAge = readSheetCaseInsensitive(
        workbook,
        "Customer Demographics"
      ) as UserAge[];

      enrichedData = mergeSheets(trips, riders, userAge);

      enrichedData = enrichedData.map((trip) => {
        const val = trip["Trip_Date_and_Time"];
        let tripDate: Date | null = null;

        if (val !== undefined && val !== null) {
          tripDate =
            typeof val === "number" ? excelDateToJSDate(val) : new Date(val);
        }

        return {
          ...trip,
          TripDateISO: tripDate?.toISOString() || null,
          TripEpoch: tripDate?.getTime() || null,
          TripYear: tripDate?.getFullYear() || null,
          TripMonth: tripDate ? tripDate.getMonth() + 1 : null,
          TripDay: tripDate?.getDate() || null,
          TripDayOfWeek: tripDate?.getDay() || null, // 0=Sun, 6=Sat
          TripHour: tripDate?.getHours() || null,
        };
      });
      vectorStore = await createVectorStore(enrichedData);

      res.json({
        message: "File processed and embeddings created",
        rows: enrichedData.length,
      });
    } catch (err: any) {
      console.error("upload error:", err);
      res.status(500).json({ error: err.message });
    }
  }
);

// --- In-memory store for chat histories ---
const userHistories = new Map<string, ChatMessageHistory>();

function getUserHistory(userId: string): ChatMessageHistory {
  if (!userHistories.has(userId)) {
    userHistories.set(userId, new ChatMessageHistory());
  }
  return userHistories.get(userId)!;
}

app.post("/chat2", async (req, res) => {
  try {
    if (!vectorStore)
      return res.status(400).json({ error: "No data uploaded yet" });

    const { question, userId } = req.body;
    if (!question || !userId)
      return res
        .status(400)
        .json({ error: "Both question and userId are required" });

    // --- Load user's chat history ---
    const history = getUserHistory(userId);

    // --- Memory for ConversationChain ---
    const memory = new BufferMemory({
      memoryKey: "history",
      inputKey: "input",
      outputKey: "response",
      chatHistory: history,
    });

    // --- Create ConversationChain with Gemini ---
    const chain = new ConversationChain({
      llm: new ChatGoogleGenerativeAI({
        model: "gemini-1.5-flash",
        apiKey: process.env.GEMINI_API_KEY,
        temperature: 0,
      }),
      memory,
      outputKey: "response",
    });

    // --- Retrieve relevant docs ---
    const retriever = vectorStore!.asRetriever({ k: 20 });
    const relevantDocs = await retriever.getRelevantDocuments(question);

    // --- Build context from retrieved docs ---
    const contextText = relevantDocs
      .map((d) => JSON.stringify(d.metadata || d.pageContent))
      .join("\n");

    // --- Include current date for reference ---
    const prompt = `You are an AI assistant with trip data. Today's date is ${new Date().toISOString()}.
      You have the following trip data:
      Context: ${contextText}

      Answer the user's question based only on this data.
      Question: ${question}`;

    // --- Call chain ---
    const response = await chain.call({ input: prompt });

    // --- Persist latest Q&A to user history ---
    history.addUserMessage(question);
    history.addAIMessage(response.response);

    res.json({ answer: response.response });
  } catch (err) {
    console.error("Error in /chat2:", err);
    res.status(500).json({ error: "Something went wrong" });
  }
});

// Health check
app.get("/", (req: Request, res: Response) => {
  res.send("Fetii Bot Backend Running 🚀");
});

app.listen(PORT, () => {
  console.log(`Backend listening at http://localhost:${PORT}`);
});
