from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import asyncio
import os
import sys

load_dotenv()

# Check if required environment variables are set
if not os.getenv("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY not found in environment variables")
    sys.exit(1)

if not os.getenv("FIRECRAWL_API_KEY"):
    print("Error: FIRECRAWL_API_KEY not found in environment variables")
    sys.exit(1)

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

server_params = StdioServerParameters(
    command="npx",
    env={
        "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY"),
    },
    args=["firecrawl-mcp"]
)

async def main():
    try:
        print("Starting MCP client...")
        async with stdio_client(server_params) as (read, write):
            print("Connected to MCP server")
            async with ClientSession(read, write) as session:
                print("Initializing session...")
                await session.initialize()
                print("Loading tools...")
                tools = await load_mcp_tools(session)
                print("Creating agent...")
                agent = create_react_agent(model, tools)

                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that can scrape websites, crawl pages, and extract data using Firecrawl tools. Think step by step and use the appropriate tools to help the user."
                    }
                ]

                print("Available Tools -", *[tool.name for tool in tools])
                print("-" * 60)
                print("Chat started! Type 'quit' to exit.")

                while True:
                    try:
                        user_input = input("\nYou: ")
                        if user_input.lower() in ["quit", "exit", "q"]:
                            print("Goodbye!")
                            break

                        messages.append({"role": "user", "content": user_input[:175000]})

                        print("Processing...")
                        agent_response = await agent.ainvoke({"messages": messages})
                        
                        ai_message = agent_response["messages"][-1].content
                        print(f"\nAgent: {ai_message}")
                        
                        # Add agent response to messages
                        messages.append({"role": "assistant", "content": ai_message})
                        
                    except KeyboardInterrupt:
                        print("\nExiting...")
                        break
                    except EOFError:
                        print("\nInput stream closed. Exiting...")
                        break
                    except Exception as e:
                        print(f"Error processing request: {e}")
                        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure firecrawl-mcp is installed: npm install -g firecrawl-mcp")
        print("2. Check that your API keys are set in .env file")
        print("3. Verify that npx is available in your PATH")

if __name__ == "__main__":
    asyncio.run(main())