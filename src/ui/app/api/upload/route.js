



import { NextResponse } from "next/server";

const { BlobServiceClient } = require('@azure/storage-blob');  

require('dotenv').config();
const { v4: uuidv4 } = require('uuid');  
  
// Generate a v4 UUID (random)  

export async function POST(req) {
    const myUUIDv4 = uuidv4();
    const blobServiceClient = new BlobServiceClient(process.env.AZURE_STORAGE_CONNECTION_STRING);  
    const blob = await req.blob();
    console.log("running wild")
    // Replace with your container name  
    const containerName = 'cobra-upload';  
    
    // Get a reference to a container  
    const containerClient = blobServiceClient.getContainerClient(containerName);  
    
    // Replace with the path to the video file you want to upload  
    // const filePath = path.join(__dirname, 'path/to/your/video.mp4');  
    // const fileName = path.basename(filePath);  
  
    // Get a block blob client  
    const blockBlobClient = containerClient.getBlockBlobClient(myUUIDv4+".mp4");  
  
    console.log('Uploading to Azure storage as blob:\n\t');  
  
    // Upload data to the blob  
    // let buf=await blob.arrayBuffer()
    // const buffer = Buffer.from(buf);
    const arr = new Uint8Array(await blob.arrayBuffer());
    const bar=arr.filter(byte => byte !== 0);
    const uploadBlobResponse = await blockBlobClient.uploadData(await blob.arrayBuffer(), bar.length,{blobHTTPHeaders:{blobContentType: 'video/mp4'}});  
    console.log("Blob was uploaded successfully. requestId: ", uploadBlobResponse.requestId); 
   return NextResponse.json({
    message: "uploaded"
  }, {
    status: 200,
  })
  }



//   export const config = {  
//     api: {  
//       bodyParser: false, // Disabling body parsing because we're using multer  
//     },  
//   };  