import { Logger } from "tslog";

import { TextLoader } from "langchain/document_loaders";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { CharacterTextSplitter } from "langchain/text_splitter";
import { PineconeStore } from "langchain/vectorstores";

import { config } from "dotenv";
import { resolve } from "path";
import { PineconeClient } from "pinecone-client";

config({
  path: ".env.local",
});

const DATA_PATH = resolve(__dirname, "../English-Quran-plain-text.txt");

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
  return splitDocuments;
};

const run = async () => {
  logger.info("Starting up");

  logger.info("loading data");
  const docs = await loadData();
  logger.info("data loaded");
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  logger.info("Initializing Pinecone client");
  const client = new PineconeClient({
    apiKey: process.env.PINECONE_API_KEY as string,
    namespace: process.env.PINECONE_ENVIRONMENT as string,
    baseUrl: process.env.PINECONE_INDEX_URL as string,
  });
  logger.info("Pinecone client initialized");

  logger.info("Initalizing and storing embeddings");
  await PineconeStore.fromDocuments(
    client,
    docs,
    embeddings,
    process.env.PINECONE_INDEX_NAME as string
  );
  logger.info("Embeddings stored");
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
