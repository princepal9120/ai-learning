import { openai } from '@ai-sdk/openai';
import { streamText, UIMessage, convertToModelMessages, tool } from 'ai';
import z from 'zod';

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  const result = streamText({
    model: openai('gpt-4o'),
    messages: convertToModelMessages(messages),
     tools: {
      db: tool({
        description: 'Call this tool to query a database',
        inputSchema: z.object({
          query: z.string().describe('The SQL query to execute'),
        }),
        execute: async ({ query }) => {
          // Simulate a database query
          const results = `Results for query: ${query}`;
          console.log(`Executing DB query: ${query}: results: ${results}`);
          return {
            results,
          };
        },
      }),
    },
  });

  return result.toUIMessageStreamResponse();
}
            temperature,
          };
        },
      }),
    },
  });

  return result.toUIMessageStreamResponse();
}