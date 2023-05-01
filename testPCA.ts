import { PCA } from "ml-pca";
import { readFileSync, writeFileSync } from "fs";
import Matrix from "ml-matrix";

async function testPca() {
  const pathToModel = "landmarksPCA.json";
  // load the saved model as json
  // Load the JSON file
  const jsonData = readFileSync(pathToModel, "utf-8");

  // Parse the JSON string into a JavaScript object
  const savedPCAMod = JSON.parse(jsonData);

  // load the model
  const pcaModel = PCA.load(savedPCAMod);
  // generate random data to transform
  const randomData: number[] = [];
  for (let i = 0; i < 1404; i++) {
    randomData.push(Math.random());
  }
  // create matrix from array
  const randomDataMatrix = Matrix.from1DArray(1, 1404, randomData);
  //   const preds = randomDataMatrix.mmul(savedPCAMod.U.transpose());
  //   console.log("_______");
  //   console.log(preds);
  // transform the data
  const transformedData = pcaModel.predict(randomDataMatrix);
  // get output as 2d list
  const transformedDataList = transformedData.to2DArray();
  console.log("Transformed data list:");
  console.log(transformedDataList);
}

testPca().then(() => console.log("Done"));
