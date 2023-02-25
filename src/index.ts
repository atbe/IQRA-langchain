import { Logger } from "tslog";

import { OpenAI } from "langchain/llms";

import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";

import { config } from "dotenv";

config({
  path: ".env.local",
});

const logger = new Logger({
  name: "logger",
});

const run = async () => {
  logger.info("Starting up");

  const model = new OpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });
  const memory = new BufferMemory();
  const chain = new ConversationChain({ llm: model, memory: memory });

  while (true) {
    // take input from CLI
    console.log("Enter input: ");

    const input = await new Promise((resolve) => {
      process.stdin.resume();
      process.stdin.setEncoding("utf8");
      process.stdin.on("data", (text) => {
        resolve(text);
      });
    });

    const res = await chain.call({ input });
    console.log({ res });
  }
};

run()
  .then(() => {
    logger.info("Finished");
    process.exit(0);
  })
  .catch((err) => {
    logger.error(err);
    process.exit(1);
  });
