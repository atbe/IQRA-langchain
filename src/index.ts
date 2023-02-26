import { Logger } from "tslog";

import { OpenAI } from "langchain/llms";

import { VectorDBQAChain } from "langchain/chains";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { PineconeStore } from "langchain/vectorstores";

import { config } from "dotenv";
import { BufferMemory, BufferWindowMemory } from "langchain/dist/memory";
import { PineconeClient } from "pinecone-client";

config({
  path: ".env.local",
});

const logger = new Logger({
  name: "logger",
});

const run = async () => {
  logger.info("Starting up");

  logger.info("loading data");
  logger.info("data loaded");
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  const client = new PineconeClient({
    apiKey: process.env.PINECONE_API_KEY as string,
    namespace: process.env.PINECONE_ENVIRONMENT as string,
    baseUrl: process.env.PINECONE_INDEX_URL as string,
  });

  const search = await PineconeStore.fromExistingIndex(
    client,
    embeddings,
    process.env.PINECONE_INDEX_NAME as string
  );
  logger.info("Chroma search initialized");
  const model = new OpenAI(
    {
      openAIApiKey: process.env.OPENAI_API_KEY,
    },
    {
      basePath: "https://oai.hconeai.com/v1",
      baseOptions: {
        headers: {
          "Helicone-Cache-Enabled": "true",
        },
      },
    }
  );

  logger.info("Initializing VectorDBQAChain");
  const chain = VectorDBQAChain.fromLLM(model, search);
  logger.info("VectorDBQAChain initialized");

  chain.memory = new BufferWindowMemory({
    k: 20,
  });

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

    const res = await chain.call({
      query: input,
    });
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
    throw err;
  });
