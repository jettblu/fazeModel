import { PCA } from "ml-pca";
import { readFileSync, writeFileSync } from "fs";

const savedPCAPathJson = "landmarksPCA.json";

// file path to the output file from extractLandmarks.py
const inputFile = "landmarksLFW.txt";
// Load the landmarks data from the saved file
const landmarksData = readFileSync(inputFile, "utf-8");

// Extract the flattened landmarks for each sample
const nameToValues: { [name: string]: number[] } = {};

landmarksData.split("\n").forEach((line: any) => {
  const [name, val1, val2, val3] = line.split(",");
  const values = [parseInt(val1), parseInt(val2), parseInt(val3)];
  if (name in nameToValues) {
    nameToValues[name].push(...values);
  } else {
    nameToValues[name] = values;
  }
});

// Convert the landmarks data to a 2D array
const landmarks = Object.values(nameToValues).slice(0, 13224);
// check dimensions of each component
// print row and row length if not 1404
let i = 0;
landmarks.forEach((row) => {
  if (row.length !== 1404) {
    console.log(row);
    console.log(i);
  }
  i++;
});

// Fit the PCA model to the landmarks data
const pca = new PCA(landmarks, {
  center: false, // Center the data
  scale: false, // Scale the data
});

console.log(pca.getExplainedVariance());

// save the PCA data
const savedPca = pca.toJSON();

// Write the PCA data to a file
writeFileSync(savedPCAPathJson, JSON.stringify(savedPca));
