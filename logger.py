"""
Logging functionality for PaperFinder LLM interactions
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

class PaperFinderLogger:
    """Logger for recording LLM interactions and agent activities"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create session log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = self.log_dir / f"session_{timestamp}.json"
        self.text_log_file = self.log_dir / f"session_{timestamp}.txt"
        self.raw_llm_file = self.log_dir / f"raw_llm_{timestamp}.jsonl"
        
        # Initialize session data
        self.session_data = {
            "session_id": timestamp,
            "start_time": datetime.now().isoformat(),
            "queries": []
        }
        
        # Initialize raw LLM log
        self._init_raw_llm_log()
        
        # Write initial session info
        self._write_session_header()
    
    def _init_raw_llm_log(self):
        """Initialize raw LLM log file"""
        with open(self.raw_llm_file, 'w', encoding='utf-8') as f:
            header = {
                "type": "session_start",
                "session_id": self.session_data['session_id'],
                "timestamp": self.session_data['start_time'],
                "note": "Raw LLM prompt/response pairs in JSONL format"
            }
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
    
    def _write_session_header(self):
        """Write session header to text log"""
        with open(self.text_log_file, 'w', encoding='utf-8') as f:
            f.write(f"🔍 PaperFinder Session Log\n")
            f.write(f"Session ID: {self.session_data['session_id']}\n")
            f.write(f"Start Time: {self.session_data['start_time']}\n")
            f.write("=" * 80 + "\n\n")
    
    def start_query(self, query: str) -> str:
        """Start logging a new query"""
        query_id = f"query_{len(self.session_data['queries']) + 1}"
        
        query_data = {
            "query_id": query_id,
            "user_query": query,
            "timestamp": datetime.now().isoformat(),
            "llm_interactions": [],
            "tool_calls": [],
            "final_result": ""
        }
        
        self.session_data["queries"].append(query_data)
        
        # Write to text log
        self._log_to_text(f"\n🚀 NEW QUERY [{query_id}]\n")
        self._log_to_text(f"Time: {query_data['timestamp']}\n")
        self._log_to_text(f"Query: {query}\n")
        self._log_to_text("-" * 50 + "\n")
        
        return query_id
    
    def log_llm_interaction(self, query_id: str, interaction_type: str, content: str, response: str = None):
        """Log LLM interaction (thinking, tool calls, responses)"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "type": interaction_type,
            "content": content,
            "response": response
        }
        
        # Find current query and add interaction
        for query in self.session_data["queries"]:
            if query["query_id"] == query_id:
                query["llm_interactions"].append(interaction)
                break
        
        # Write to text log
        if interaction_type == "thinking":
            self._log_to_text(f"🤔 THINKING: {content}\n")
        elif interaction_type == "tool_call":
            self._log_to_text(f"🔧 TOOL CALL: {content}\n")
        elif interaction_type == "tool_response":
            self._log_to_text(f"✅ TOOL RESPONSE: {content[:200]}{'...' if len(content) > 200 else ''}\n")
        elif interaction_type == "llm_response":
            self._log_to_text(f"🤖 LLM RESPONSE: {content}\n")
        
        self._save_session()
    
    def log_tool_call(self, query_id: str, tool_name: str, args: Dict[str, Any], result: str):
        """Log tool call details"""
        tool_call = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "arguments": args,
            "result": result
        }
        
        # Find current query and add tool call
        for query in self.session_data["queries"]:
            if query["query_id"] == query_id:
                query["tool_calls"].append(tool_call)
                break
        
        # Write to text log
        self._log_to_text(f"🔧 TOOL: {tool_name}\n")
        if args:
            for key, value in args.items():
                self._log_to_text(f"   {key}: {value}\n")
        result_preview = result[:300] + "..." if len(result) > 300 else result
        self._log_to_text(f"   Result: {result_preview}\n")
        
        self._save_session()
    
    def log_raw_llm_interaction(self, query_id: str, messages: List[Dict], response: str, model: str = "gpt-4o-mini"):
        """Log raw LLM prompt and response"""
        try:
            raw_entry = {
                "type": "llm_interaction",
                "timestamp": datetime.now().isoformat(),
                "query_id": query_id,
                "model": model,
                "messages": messages,
                "response": response
            }
            
            with open(self.raw_llm_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(raw_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            import traceback
            error_msg = f"Failed to log raw LLM interaction: {e}\nTraceback:\n{traceback.format_exc()}"
            print(error_msg)
            try:
                error_file = self.log_dir / "logging_error.log"
                with open(error_file, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.now().isoformat()}: {error_msg}\n\n")
            except:
                pass
    
    def log_final_result(self, query_id: str, result: str):
        """Log final result for a query"""
        # Find current query and set final result
        for query in self.session_data["queries"]:
            if query["query_id"] == query_id:
                query["final_result"] = result
                query["end_time"] = datetime.now().isoformat()
                break
        
        # Write to text log
        self._log_to_text(f"\n✅ FINAL RESULT:\n")
        self._log_to_text(result + "\n")
        self._log_to_text("=" * 80 + "\n")
        
        self._save_session()
    
    def _log_to_text(self, message: str):
        """Write message to text log file"""
        with open(self.text_log_file, 'a', encoding='utf-8') as f:
            f.write(message)
    
    def _save_session(self):
        """Save session data to JSON file"""
        try:
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            import traceback
            error_msg = f"Failed to save session: {e}\nTraceback:\n{traceback.format_exc()}"
            print(error_msg)
            # Try to save error to a separate file
            try:
                error_file = self.log_dir / "save_error.log"
                with open(error_file, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.now().isoformat()}: {error_msg}\n\n")
            except:
                pass
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        return {
            "session_id": self.session_data["session_id"],
            "start_time": self.session_data["start_time"],
            "total_queries": len(self.session_data["queries"]),
            "log_files": {
                "json": str(self.session_file),
                "text": str(self.text_log_file),
                "raw_llm": str(self.raw_llm_file)
            }
        }
    
    def close_session(self):
        """Close logging session"""
        self.session_data["end_time"] = datetime.now().isoformat()
        self._save_session()
        
        self._log_to_text(f"\n🏁 SESSION ENDED\n")
        self._log_to_text(f"End Time: {self.session_data['end_time']}\n")
        self._log_to_text(f"Total Queries: {len(self.session_data['queries'])}\n")

# Global logger instance
logger = None

def get_logger() -> PaperFinderLogger:
    """Get or create global logger instance"""
    global logger
    if logger is None:
        logger = PaperFinderLogger()
    return logger

def init_logger(log_dir: str = "./logs") -> PaperFinderLogger:
    """Initialize logger with custom directory"""
    global logger
    logger = PaperFinderLogger(log_dir)
    return logger