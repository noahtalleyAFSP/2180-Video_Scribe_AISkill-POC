### Welcome to COBRA's Frontend User Interface

### Instructions on usage are below.

## Instructions

    - Navigate to the ui folder
    - Open up the public folder
    - put your video in the public folder and name it "video.mp4"
    - navigate into app/data to see the ActionSummary and Chapter Analysis
    - Take the ActionSummary and ChapterAnalysis generated in cobrapy and put them into the data folder
    - go the the .env file, and populate the parameters with the appropriate values, most importantly, put in the name of the search index created out of the ActionSummary
    - Go to the compnents/player file and change the Topic Analysis to any value you want
    - proceed to deployment

## Deployment

    - npm i (install dependencies)
    - For local deploy, navigate to the ui folder and "npm run dev"
    - for docker deploy, navigate to the ui folder and docker build . -t cobra-{your_name}
    - after docker build then docker run with a port map of 3000:3000
