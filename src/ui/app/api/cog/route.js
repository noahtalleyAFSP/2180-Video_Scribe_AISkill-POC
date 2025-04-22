import { NextResponse } from "next/server";
import { NextApiResponse } from "next";
const { SearchClient, AzureKeyCredential } = require("@azure/search-documents");
require('dotenv').config();

export async function POST(req) {
  const client = new SearchClient(
    process.env.SEARCH_ENDPOINT,
    process.env.INDEX_NAME,
    new AzureKeyCredential(process.env.SEARCH_API_KEY)
  );
    // Extract the `messages` from the body of the request
    const { messages } = await req.json()
    try{
      const searchResults = await client.search(messages,{queryType:"semantic",queryLanguage:"en-us",top:5,semanticConfiguration:"sem"})
      let docs=[]
    let count=1
    for await (const result of searchResults.results) {
      if(count>5) break
      count+=1
      docs.push(result.document)
    }
   return NextResponse.json({
    message: docs
  }, {
    status: 200,
  })
    }
    catch(e){
      let the_Data=JSON.parse(e.message).value
      let counts=1
      let docs=[]
      for await (const result of the_Data) {
        if(counts>5) break
        counts+=1
        docs.push(result)
        
      }
    
      
    // let count=1
    // for await (const result of searchResults.results) {
    //   if(count>5) break
    //   count+=1
    //   docs.push(result.document)
    // }
   return NextResponse.json({
    message: docs
  }, {
    status: 200,
  })
      
    }
    
    
    
  }