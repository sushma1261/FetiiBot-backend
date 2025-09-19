// src/middleware/auth.ts
import { NextFunction, Request, Response } from "express";
import { verifyToken } from "../config/firebase";

export const authenticate = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  const authEnabled = process.env.AUTH_ENABLED === "true";
  if (!authEnabled) {
    console.log("Authentication disabled - skipping auth middleware");
    return next();
  }
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return res.status(401).json({ message: "No token provided" });
  }

  const token = authHeader.split(" ")[1];
  const decoded = await verifyToken(token);

  if (!decoded) {
    return res.status(401).json({ message: "Invalid token" });
  }

  // Attach user info to request
  (req as any).user = decoded;
  console.log("Authenticated user:", decoded);
  next();
};
