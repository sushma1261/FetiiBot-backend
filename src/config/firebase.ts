import * as dotenv from "dotenv";
import * as admin from "firebase-admin";

dotenv.config(); // Load .env.local

if (!admin.apps.length) {
  admin.initializeApp({
    credential: admin.credential.cert({
      projectId: process.env.FIREBASE_PROJECT_ID,
      clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
      privateKey: process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, "\n"),
    }),
  });
}

export const verifyToken = async (token: string) => {
  try {
    const decodedToken = await admin.auth().verifyIdToken(token);
    return decodedToken;
  } catch (error) {
    console.error("Error verifying Firebase ID token:", error);
    return null;
  }
};

// import * as admin from "firebase-admin";

// if (!admin.apps.length) {
//   admin.initializeApp({
//     credential: admin.credential.applicationDefault(),
//   });
// }

// export const verifyToken = async (token: string) => {
//   try {
//     console.log("Verifying token:", token);
//     const decodedToken = await admin.auth().verifyIdToken(token);
//     return decodedToken;
//   } catch (error) {
//     console.error("Error verifying Firebase ID token:", error);
//     return null;
//   }
// };
