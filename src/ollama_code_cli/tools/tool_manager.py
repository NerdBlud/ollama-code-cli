

import json
import os
import sys
import subprocess
import tempfile
import shutil
import fnmatch
import time
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.prompt import Confirm
from rich.panel import Panel


# ============================================================
# Tool Manager
# ============================================================

class ToolManager:
    """Main tool manager for the Ollama Code CLI — all tools live here."""

    # ---------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------

    def __init__(
        self,
        require_permission: bool = True,
        rate_limit_ms: int = 150,   # small safety delay between dangerous ops
    ):
        self.console = Console()
        self.require_permission = require_permission
        self.rate_limit_ms = rate_limit_ms
        self.tools = self._define_tools()
        self.last_tool_call_time = 0


    # ---------------------------------------------------------
    # Tool Definitions
    # ---------------------------------------------------------

    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """Define all tools and their schemas."""
        return {

            # -------------------------------------------------
            # BASIC FILE OPERATIONS
            # -------------------------------------------------

            "read_file": {
                "function": self._read_file,
                "description": "Read the contents of a file.",
                "requires_permission": False,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string"},
                        "max_bytes": {
                            "type": "integer",
                            "description": "Optional limit to prevent huge reads."
                        }
                    },
                    "required": ["filepath"],
                },
            },

            "write_file": {
                "function": self._write_file,
                "description": "Write text content to a file.",
                "requires_permission": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string"},
                        "content": {"type": "string"},
                        "append": {
                            "type": "boolean",
                            "description": "If true, append instead of overwrite."
                        },
                    },
                    "required": ["filepath", "content"],
                },
            },

            "list_files": {
                "function": self._list_files,
                "description": "List files in a directory.",
                "requires_permission": False,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "default": "."},
                        "pattern": {
                            "type": "string",
                            "description": "Optional glob filter (e.g. *.py)"
                        }
                    }
                },
            },

            "file_info": {
                "function": self._file_info,
                "description": "Get metadata about a file.",
                "requires_permission": False,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string"}
                    },
                    "required": ["filepath"],
                }
            },

            "delete_file": {
                "function": self._delete_file,
                "description": "Delete a file safely.",
                "requires_permission": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string"}
                    },
                    "required": ["filepath"],
                },
            },

            "make_directory": {
                "function": self._make_directory,
                "description": "Create a directory (including nested).",
                "requires_permission": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"],
                },
            },


            # -------------------------------------------------
            # CODE EXECUTION
            # -------------------------------------------------

            "execute_code": {
                "function": self._execute_code,
                "description": "Execute code in a sandboxed subprocess.",
                "requires_permission": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "language": {"type": "string", "default": "python"},
                        "stdin": {"type": "string"},
                        "timeout": {"type": "integer", "default": 30}
                    },
                    "required": ["code"],
                }
            },

            "run_python_file": {
                "function": self._run_python_file,
                "description": "Execute an existing Python file.",
                "requires_permission": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string"},
                        "timeout": {"type": "integer", "default": 30}
                    },
                    "required": ["filepath"],
                }
            },


            # -------------------------------------------------
            # COMMAND EXECUTION
            # -------------------------------------------------

            "run_command": {
                "function": self._run_command,
                "description": "Execute a shell command.",
                "requires_permission": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer", "default": 30}
                    },
                    "required": ["command"],
                }
            },
        }


    # ---------------------------------------------------------
    # Permission + Safety Layer
    # ---------------------------------------------------------

    def _safety_wait(self):
        """Apply a rate-limit delay to avoid rapid chained tool calls."""
        now = time.time()
        diff = (now - self.last_tool_call_time) * 1000
        if diff < self.rate_limit_ms:
            time.sleep((self.rate_limit_ms - diff) / 1000)
        self.last_tool_call_time = time.time()

    def _ask_permission(self, tool: str, description: str) -> bool:
        if not self.require_permission:
            return True
        panel = Panel(
            f"[bold yellow]⚠️ Tool Requires Permission[/bold yellow]\n\n"
            f"[bold]Tool:[/bold] {tool}\n"
            f"[bold]Reason:[/bold] {description}",
            title="Security Check",
            border_style="yellow",
        )
        self.console.print(panel)
        return Confirm.ask("[bold blue]Proceed?[/bold blue]", default=False)

    def _safe_path(self, path: str) -> str:
        """Normalize + prevent nonsense paths."""
        return os.path.abspath(os.path.expanduser(path))


    # ---------------------------------------------------------
    # File Tools
    # ---------------------------------------------------------

    def _read_file(self, filepath: str, max_bytes: Optional[int] = None) -> Dict[str, Any]:
        try:
            path = self._safe_path(filepath)
            with open(path, "rb") as f:
                data = f.read(max_bytes) if max_bytes else f.read()
            return {"status": "success", "content": data.decode("utf-8", errors="replace")}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _write_file(self, filepath: str, content: str, append: bool = False) -> Dict[str, Any]:
        path = self._safe_path(filepath)

        action = "Append to" if append else "Write to"
        if not self._ask_permission("write_file", f"{action} file: {path}"):
            return {"status": "cancelled"}

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            mode = "a" if append else "w"
            with open(path, mode, encoding="utf-8") as f:
                f.write(content)
            return {"status": "success", "message": f"{action} succeeded"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _list_files(self, path: str = ".", pattern: str = None) -> Dict[str, Any]:
        try:
            root = self._safe_path(path)
            files = os.listdir(root)
            if pattern:
                files = [f for f in files if fnmatch.fnmatch(f, pattern)]
            return {"status": "success", "files": files}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _file_info(self, filepath: str) -> Dict[str, Any]:
        try:
            path = self._safe_path(filepath)
            stat = os.stat(path)
            return {
                "status": "success",
                "info": {
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "created": stat.st_ctime,
                    "is_file": os.path.isfile(path),
                    "is_dir": os.path.isdir(path),
                },
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _delete_file(self, filepath: str) -> Dict[str, Any]:
        path = self._safe_path(filepath)
        if not self._ask_permission("delete_file", f"Remove file: {path}"):
            return {"status": "cancelled"}

        try:
            os.remove(path)
            return {"status": "success", "message": "File deleted"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _make_directory(self, path: str) -> Dict[str, Any]:
        real = self._safe_path(path)
        if not self._ask_permission("make_directory", f"Create directory: {real}"):
            return {"status": "cancelled"}
        try:
            os.makedirs(real, exist_ok=True)
            return {"status": "success", "message": "Directory created"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


    # ---------------------------------------------------------
    # Code Execution
    # ---------------------------------------------------------

    def _execute_code(
        self,
        code: str,
        language: str = "python",
        stdin: Optional[str] = None,
        timeout: int = 30,
    ) -> Dict[str, Any]:

        code_preview = code[:120].replace("\n", " ") + ("..." if len(code) > 120 else "")
        if not self._ask_permission("execute_code", f"Run {language} code:\n{code_preview}"):
            return {"status": "cancelled"}

        if language != "python":
            return {"status": "error", "message": f"Unsupported language: {language}"}

        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_path = f.name

            proc = subprocess.run(
                [sys.executable, temp_path],
                input=stdin,
                text=True,
                capture_output=True,
                timeout=timeout,
            )

            return {
                "status": "success" if proc.returncode == 0 else "error",
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
            }

        except subprocess.TimeoutExpired:
            return {"status": "error", "message": f"Execution exceeded {timeout}s"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass


    def _run_python_file(self, filepath: str, timeout: int = 30) -> Dict[str, Any]:
        path = self._safe_path(filepath)

        if not self._ask_permission("run_python_file", f"Execute: {path}"):
            return {"status": "cancelled"}

        if not os.path.exists(path):
            return {"status": "error", "message": "File not found"}

        try:
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "status": "success" if proc.returncode == 0 else "error",
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
            }

        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "Timeout"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


    # ---------------------------------------------------------
    # Shell Commands
    # ---------------------------------------------------------

    def _run_command(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        if not self._ask_permission("run_command", f"Shell command:\n{command}"):
            return {"status": "cancelled"}

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "status": "success" if proc.returncode == 0 else "error",
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "Timeout"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


    # ---------------------------------------------------------
    # Ollama API Interface
    # ---------------------------------------------------------

    def get_tools_for_ollama(self) -> List[dict]:
        """Return tool definitions in Ollama-function format."""
        out = []
        for name, t in self.tools.items():
            out.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": t["description"],
                    "parameters": t["parameters"],
                }
            })
        return out


    def handle_tool_calls(self, tool_calls: List[dict]) -> List[dict]:
        """Execute LLM tool calls sequentially with safety + validation."""
        results = []
        for call in tool_calls:

            name = call.get("function", {}).get("name")
            args = call.get("function", {}).get("arguments", {})

            self._safety_wait()

            # Parse arguments JSON or dict
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    results.append({
                        "role": "tool",
                        "name": name,
                        "content": json.dumps({"status": "error", "message": "Invalid JSON arguments"}),
                    })
                    continue

            if name not in self.tools:
                results.append({
                    "role": "tool",
                    "name": name,
                    "content": json.dumps({"status": "error", "message": f"Unknown tool: {name}"})
                })
                continue

            try:
                out = self.tools[name]["function"](**args)
                results.append({
                    "role": "tool",
                    "name": name,
                    "content": json.dumps(out),
                })
            except TypeError as e:
                results.append({
                    "role": "tool",
                    "name": name,
                    "content": json.dumps({"status": "error", "message": f"Bad arguments: {str(e)}"}),
                })
            except Exception as e:
                results.append({
                    "role": "tool",
                    "name": name,
                    "content": json.dumps({"status": "error", "message": str(e)}),
                })

        return results
