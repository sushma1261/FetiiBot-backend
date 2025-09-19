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
    console.log("Trip:", trip);
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
      console.log(workbook.SheetNames);
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
      // let count = 0;
      // --- Convert Trip Date to Date and Epoch ---
      // enrichedData = enrichedData.map((trip) => {
      //   let tripDateObj: Date | null = null;

      //   const val = trip["Trip_Date_and_Time"];
      //   if (val !== undefined && val !== null) {
      //     if (typeof val === "number") {
      //       tripDateObj = excelDateToJSDate(val);
      //     } else {
      //       tripDateObj = new Date(val); // fallback for proper string dates
      //     }
      //   }

      //   const TripDateISO = tripDateObj ? tripDateObj.toISOString() : null;
      //   const TripEpoch = tripDateObj ? tripDateObj.getTime() : null;
      //   console.log({ ...trip, TripDateISO, TripEpoch });
      //   return {
      //     ...trip,
      //     TripDateISO,
      //     TripEpoch,
      //   };
      // });

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
      // console.log("Found NaN date", count);
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

// --- /chat2 route ---
// app.post("/chat2", async (req, res) => {
//   try {
//     if (!vectorStore)
//       return res.status(400).json({ error: "No data uploaded yet" });

//     const { question, userId } = req.body;
//     if (!question || !userId)
//       return res
//         .status(400)
//         .json({ error: "Both question and userId are required" });

//     // --- Optional pre-filter example ---
//     let filteredData = enrichedData;
//     if (/age\s*18-24/i.test(question)) {
//       filteredData = enrichedData.filter((d) => d.Age >= 18 && d.Age <= 24);
//     }

//     // --- Recreate vector store if filtered ---
//     const filteredVectorStore = await createVectorStore(filteredData);

//     // --- Load user's chat history ---
//     const history = getUserHistory(userId);

//     // --- Memory for ConversationChain ---
//     const memory = new BufferMemory({
//       memoryKey: "history",
//       inputKey: "input",
//       outputKey: "response",
//       chatHistory: history,
//     });

//     // --- Create ConversationChain with Gemini ---
//     const chain = new ConversationChain({
//       llm: new ChatGoogleGenerativeAI({
//         model: "gemini-1.5-flash",
//         apiKey: process.env.GEMINI_API_KEY,
//         temperature: 0,
//       }),
//       memory,
//       outputKey: "response",
//     });

//     // --- Retrieve relevant docs ---
//     const retriever = filteredVectorStore.asRetriever();
//     const relevantDocs = await retriever.getRelevantDocuments(question);

//     // --- Build context from retrieved docs ---
//     const contextText = relevantDocs
//       .map((d) => JSON.stringify(d.metadata || d.pageContent))
//       .join("\n");

//     const finalInput = `${contextText}\n\nUser question: ${question}`;
// const prompt = new PromptTemplate({
//       template: `
// You are an expert assistant analyzing trip data.
// Each trip has the following fields: Trip_ID, Pick_Up_Latitude, Pick_Up_Longitude,
// Drop_Off_Latitude, Drop_Off_Longitude, Pick_Up_Address, Drop_Off_Address,
// TripDateISO, TripEpoch, Total_Passengers, Age, and checkedInUserID.

// Use the context below to answer the user's question accurately.
// Do not make assumptions outside the given data.

// Context:
// {context}

// Question: {question}

// Answer:
//       `,
//       inputVariables: ["context", "question"],
//     });

//     const finalPrompt = await prompt.format({
//       context: contextText,
//       question,
//     });
//     // --- Call chain ---
//     const response = await chain.call({ input: finalInput });

//     // --- Persist latest Q&A to user history ---
//     history.addUserMessage(question);
//     history.addAIMessage(response.response);

//     res.json({ answer: response.response });
//   } catch (err) {
//     console.error("Error in /chat2:", err);
//     res.status(500).json({ error: "Something went wrong" });
//   }
// });

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
    const retriever = vectorStore!.asRetriever();
    const relevantDocs = await retriever.getRelevantDocuments(question);

    // --- Build context from retrieved docs ---
    const contextText = relevantDocs
      .map((d) => JSON.stringify(d.metadata || d.pageContent))
      .join("\n");
    const currentDate = new Date().toISOString(); // current date in ISO

    // --- Prompt Template ---
    //     const prompt = new PromptTemplate({
    //       template: `
    // You are an expert assistant analyzing trip data.
    // Each trip has the following fields: Trip_ID, Pick_Up_Latitude, Pick_Up_Longitude,
    // Drop_Off_Latitude, Drop_Off_Longitude, Pick_Up_Address, Drop_Off_Address,
    // TripDateISO, TripEpoch, Total_Passengers, Age, and checkedInUserID.

    // Use the context below to answer the user's question accurately.
    // Do not make assumptions outside the given data.
    // Use the Current date: ${currentDate} so that you can answer questions for last month or laste week, etc.
    // Context:
    // {context}

    // Question: {question}
    //       `,
    //       inputVariables: ["context", "question"],
    //     });

    //     const finalPrompt = await prompt.format({
    //       context: contextText,
    //       question,
    //     });

    // const tableText = relevantDocs
    //   .map((d) => {
    //     const m = d.metadata;
    //     return `Trip_ID: ${m.Trip_ID} | Drop_Off_Address: ${m.Drop_Off_Address} | Pick_Up_Address: ${m.Pick_Up_Address} | Age: ${m.Age} | Total_Passengers: ${m.Total_Passengers} | TripDayOfWeek: ${m.TripDayOfWeek} | TripHour: ${m.TripHour}`;
    //   })
    //   .join("\n");

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

app.listen(PORT, () => {
  console.log(`Backend listening at http://localhost:${PORT}`);
});
