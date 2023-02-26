import { Logger } from "tslog";

import { OpenAI } from "langchain/llms";

import { ConversationChain, VectorDBQAChain } from "langchain/chains";
import { TextLoader } from "langchain/document_loaders";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { BufferMemory } from "langchain/memory";
import { CharacterTextSplitter } from "langchain/text_splitter";
import { Chroma } from "langchain/vectorstores";
import { HNSWLib } from "langchain/vectorstores";

import { config } from "dotenv";
import { resolve } from "path";

config({
  path: ".env.local",
});

const DATA_PATH = resolve(__dirname, "../English-Quran-plain-text.txt");
const DOCUMENT_LIMIT = 10;

const logger = new Logger({
  name: "logger",
});

const loadData = async () => {
  const loader = new TextLoader(DATA_PATH);
  const documents = await loader.load();
  const textSplitter = new CharacterTextSplitter({
    separator: "\n",
  });

  const splitDocuments = textSplitter.splitDocuments(documents);
  return splitDocuments.slice(0, DOCUMENT_LIMIT);
};

const runConversationLoop = async () => {
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

const run = async () => {
  logger.info("Starting up");

  logger.info("loading data");
  const docs = await loadData();
  logger.info("data loaded");
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  logger.info("Initializing Chroma search");
  //   const search = await Chroma.fromDocuments(docs, embeddings);
  const search = await HNSWLib.fromDocuments(docs, embeddings);
  logger.info("Chroma search initialized");
  const model = new OpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });
  logger.info("Initializing VectorDBQAChain");
  const chain = VectorDBQAChain.fromLLM(model, search);
  logger.info("VectorDBQAChain initialized");

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
      input_documents: docs,
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

    process.exit(1);
  });
