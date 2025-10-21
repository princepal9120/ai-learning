import { db } from '@/db/db';
import { openai } from '@ai-sdk/openai';
import { streamText, UIMessage, convertToModelMessages, tool, stepCountIs } from 'ai';
import z from 'zod';

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
    const { messages }: { messages: UIMessage[] } = await req.json();

    const SYSTEM_PROMPT = `You are an expert SQL assistant that helps users to query their database using natural language.

    ${new Date().toLocaleString('sv-SE')}
    You have access to following tools:
    1. db tool - call this tool to query the database.
    2. schema tool - call this tool to get the database schema which will help you to write sql query.

Rules:
- Generate ONLY SELECT queries (no INSERT, UPDATE, DELETE, DROP)
- Always use the schema provided by the schema tool
- Pass in valid SQL syntax in db tool.
- IMPORTANT: To query database call db tool, Don't return just SQL query.

Always respond in a helpful, conversational tone while being technically accurate.`;

    const result = streamText({
        model: openai('gpt-5-nano-2025-08-07'),
        messages: convertToModelMessages(messages),
        system: SYSTEM_PROMPT,
        stopWhen: stepCountIs(5),
        tools: {
            schema: tool({
                description: 'Call this tool to get database schema information.',
                inputSchema: z.object({}),
                execute: async () => {
                    return `CREATE TABLE products (
	id integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	name text NOT NULL,
	category text NOT NULL,
	price real NOT NULL,
	stock integer DEFAULT 0 NOT NULL,
	created_at text DEFAULT CURRENT_TIMESTAMP
)

CREATE TABLE sales (
	id integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	product_id integer NOT NULL,
	quantity integer NOT NULL,
	total_amount real NOT NULL,
	sale_date text DEFAULT CURRENT_TIMESTAMP,
	customer_name text NOT NULL,
	region text NOT NULL,
	FOREIGN KEY (product_id) REFERENCES products(id) ON UPDATE no action ON DELETE no action
)`;
                },
            }),
            db: tool({
                description: 'Call this tool to query a database.',
                inputSchema: z.object({
                    query: z.string().describe('The SQL query to be ran.'),
                }),
                execute: async ({ query }) => {
                    console.log('Query', query);
                    // Important: make sure you sanitize / validate (somehow) check the query
                    // string search [delete, update] -> Guardrails
                    return await db.run(query);
                },
            }),
        },
    });

    return result.toUIMessageStreamResponse();
}