# Customer-Support-Bot with LangGraph 🧠
Building a Customer Support Chatbot with LangGraph.

This is an adaptation of the customer support bot of the LangGraph docs. So props to them. Here is the [link](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support) where is all explained.

✈️ This project demonstrates how to build an airline customer support bot using LangGraph, an AI conversational platform. The bot can assist users in researching and making travel arrangements, including flights, hotels, car rentals, and excursions.

👀 **Browse through the 4 branches to find the bot that best suits your needs. This is the most complex one. However, more complexity doesn't always mean better.**

![Descripción de la imagen](assets/Specialized-Workflows.png)

## Environment Variables
To run this project, you will need to add the following environment variables to your .env file:
```
OPENAI_API_KEY=<your-openai-api-key>
```
How to get you OpenAI API Key https://platform.openai.com/account/api-key
```
TAVILY_API_KEY=<your-tavily-api-key>
```
How to get you Tavily API Key https://docs.mindmac.app/how-to.../internet-browsing/get-tavily-key

## Run Locally
Clone the project
```
git clone https://github.com/PrMestizo/Customer-Support-Bot.git
```

Go to the project directory
```
cd Customer-Support-Bot
```

Install dependencies
```
poetry install
```

Activate the virtual environment
```
poetry shell
```

Run database.py to get the sql files
```
python database.py
```

Last one! Run graph.py and see the power os the bot.
```
python graph.py
```

**Frontend on process** Let me cook.
