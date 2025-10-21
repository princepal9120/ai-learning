import dotenv from "dotenv";
import type { Config } from "drizzle-kit";
dotenv.config({ path: '.env.local' });
export default {
   schema: './src/db/schema.ts',
   out: './src/db/migrations',
   dialect: "turso",
   dbCredentials: {
    url: process.env.TURSO_DATABASE_URL!,
    authToken: process.env.TURSO_AUTH_TOKEN,
  },
} satisfies Config;