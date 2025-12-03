import os
import json
import click
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from ollama import Client
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from prompt_toolkit import prompt as pt_prompt
from yaspin import yaspin


# ======================================================================================
#  TOOL MANAGER
# ======================================================================================

@dataclass
class ToolManager:
    """Manages all available tools and tool invocation."""
    available_tools: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "run_code": {
            "type": "function",
            "function": {
                "name": "run_code",
                "description": "Executes Python code safely and returns stdout or error.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        },
        "read_file": {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Reads content from a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        "write_file": {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Writes text to a file, overwriting existing content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        "list_files": {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "Lists files in a directory.",
                "parameters": {
                    "type": "object",
                    "properties": {"directory": {"type": "string"}},
                    "required": ["directory"],
                },
            },
        },
        "create_dir": {
            "type": "function",
            "function": {
                "name": "create_dir",
                "description": "Creates a directory.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        "search_files": {
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Searches for files matching a query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string"},
                        "query": {"type": "string"},
                    },
                    "required": ["directory", "query"],
                },
            },
        },
    })

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return the list of tool definitions."""
        return list(self.available_tools.values())

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a tool definition by name."""
        return self.available_tools.get(name)


# ======================================================================================
#  MAIN CLI CLASS WITH ALL PATCHES
# ======================================================================================

class OllamaCodeCLI:
    """Enhanced monolithic CLI with streaming, summarization, smarter intent, granular permissions."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.client = Client()
        self.tool_manager = ToolManager()

        self.console = Console()
        self.streaming = False

        self.conversation_history: List[Dict[str, str]] = []

        # granular per-tool permissions
        self.tool_permissions = {
            "write_file": False,
            "run_code": True,
            "read_file": True,
            "list_files": True,
            "create_dir": True,
            "search_files": True,
        }

    # ==========================================================================
    #  UTILITY: SUMMARIZATION
    # ==========================================================================

    def _maybe_summarize(self):
        MAX_MESSAGES = 22
        if len(self.conversation_history) < MAX_MESSAGES:
            return

        old = self.conversation_history[:-8]

        req = [
            {"role": "system", "content": "Summarize the conversation in 6 compact bullet points."},
            {"role": "user", "content": str(old)},
        ]

        with yaspin(text="Summarizing conversation…", color="cyan"):
            summary = self.client.chat(model=self.model, messages=req)["message"]["content"]

        self.conversation_history = [
            {"role": "system", "content": f"Summary: {summary}"},
            *self.conversation_history[-8:],
        ]

    # ==========================================================================
    #  UTILITY: SMARTER TOOL INTENT
    # ==========================================================================

    def _validate_tool_usage(self, llm_message, user_input):
        ll_lower = llm_message.lower()
        user_lower = user_input.lower()

        if '"tool_call"' in ll_lower or '"tool":' in ll_lower:
            return None

        semantic = {
            "execute": ["run", "execute", "test this", "try this", "run this"],
            "files": ["save", "write file", "create file", "put this into"],
        }

        missing = []

        if any(x in user_lower for x in semantic["execute"]):
            if "run_code" not in ll_lower:
                missing.append("run_code")

        if any(x in user_lower for x in semantic["files"]):
            if "write_file" not in ll_lower:
                missing.append("write_file")

        return missing if missing else None

    # ==========================================================================
    #  PERMISSION HANDLING
    # ==========================================================================

    def _ask_permission(self, tool_name: str, fields: dict) -> bool:
        if self.tool_permissions.get(tool_name):
            return True

        lst = ", ".join(fields.keys())
        self.console.print(f"[yellow]Tool wants: {tool_name} ({lst})[/yellow]")

        choice = Prompt.ask("Allow? (y/n/always)", choices=["y", "n", "always"])
        if choice == "always":
            self.tool_permissions[tool_name] = True
            return True

        return choice == "y"

    # ==========================================================================
    #  SENDING TO MODEL (STREAMING INCLUDED)
    # ==========================================================================

    def send_to_ollama(self, user_input: str, tools: Optional[List[dict]] = None, stream: bool = False):
        self._maybe_summarize()

        self.conversation_history.append({"role": "user", "content": user_input})

        included = tools if tools else []

        if stream:
            response = self.client.chat(
                model=self.model,
                messages=self.conversation_history,
                tools=included,
                stream=True,
            )

            streamed = ""
            for chunk in response:
                msg = chunk.get("message", {}).get("content", "")
                if msg:
                    streamed += msg
                    self.console.print(msg, end="")

            self.console.print()
            return False, streamed, None

        with yaspin(text="Thinking...", color="green"):
            response = self.client.chat(
                model=self.model,
                messages=self.conversation_history,
                tools=included,
            )

        ai_msg = response["message"]["content"]
        tool_call = response["message"].get("tool_call")

        missing = self._validate_tool_usage(ai_msg, user_input)

        return bool(tool_call), ai_msg, tool_call if tool_call else missing

    # ==========================================================================
    #  TOOL EXECUTION
    # ==========================================================================

    def run_tool(self, tool_name: str, arguments: Dict[str, Any]):
        if not self._ask_permission(tool_name, arguments):
            return "Permission denied."

        try:
            if tool_name == "run_code":
                return self._run_code(arguments["code"])
            if tool_name == "read_file":
                return self._read_file(arguments["path"])
            if tool_name == "write_file":
                return self._write_file(arguments["path"], arguments["content"])
            if tool_name == "list_files":
                return self._list_files(arguments["directory"])
            if tool_name == "create_dir":
                return self._create_dir(arguments["path"])
            if tool_name == "search_files":
                return self._search_files(arguments["directory"], arguments["query"])
        except Exception as e:
            return f"Error: {e}"

        return "Unknown tool."

    # ==========================================================================
    #  TOOL BACKENDS
    # ==========================================================================

    def _run_code(self, code: str):
        try:
            local = {}
            exec(code, {}, local)
            return str(local) or "Code executed with no output."
        except Exception as e:
            return f"Runtime error: {e}"

    def _read_file(self, path: str):
        if not os.path.exists(path):
            return "File does not exist."
        with open(path, "r") as f:
            return f.read()

    def _write_file(self, path: str, content: str):
        with open(path, "w") as f:
            f.write(content)
        return f"Wrote file: {path}"

    def _list_files(self, directory: str):
        if not os.path.isdir(directory):
            return "Directory not found."
        return "\n".join(os.listdir(directory))

    def _create_dir(self, path: str):
        os.makedirs(path, exist_ok=True)
        return f"Directory created: {path}"

    def _search_files(self, directory: str, query: str):
        if not os.path.isdir(directory):
            return "Directory not found."
        matches = [f for f in os.listdir(directory) if query in f]
        return "\n".join(matches) if matches else "No matches found."

    # ==========================================================================
    #  INTERACTIVE LOOP
    # ==========================================================================

    def run_interactive(self):
        self.console.print(Panel("Ollama Code CLI", style="bold cyan"))
        while True:
            user_input = pt_prompt("> ")

            if user_input.strip() == "exit":
                break

            if self.streaming:
                self.console.print("[dim]Streaming...[/dim]")
                self.send_to_ollama(
                    user_input,
                    tools=self.tool_manager.get_tools(),
                    stream=True,
                )
                continue

            used, llm_msg, tool_or_missing = self.send_to_ollama(
                user_input, tools=self.tool_manager.get_tools()
            )

            if used and isinstance(tool_or_missing, dict):
                tool_name = tool_or_missing["function"]
                args = json.loads(tool_or_missing["arguments"])
                result = self.run_tool(tool_name, args)
                self.console.print(Panel(str(result), title=f"Tool: {tool_name}"))
                continue

            if isinstance(tool_or_missing, list):
                self.console.print(f"[yellow]LLM forgot tools: {tool_or_missing}[/yellow]")

            self.console.print(Markdown(llm_msg))


# ======================================================================================
#  CLI COMMANDS
# ======================================================================================

@click.group()
def cli():
    pass


@cli.command()
@click.option("--model", default="llama3.2", help="Model name.")
def chat(model):
    OllamaCodeCLI(model).run_interactive()


if __name__ == "__main__":
    cli()
