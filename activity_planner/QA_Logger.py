import json
import os
from typing import Dict, List, Any

class QALogger:
    """Logs questions, answers, and retrieved context to a JSON file."""
    
    def __init__(self, log_file_path: str = None):
        """
        Initialize QA Logger
        
        Args:
            log_file_path: Path to save the QA log JSON file
                          Default: qa_context_log.json in project root
        """
        if log_file_path is None:
            # Get the project root (parent of activity_planner)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            log_file_path = os.path.join(project_root, "qa_context_log.json")
        
        self.log_file_path = log_file_path
        self._ensure_log_file_exists()
    
    def _ensure_log_file_exists(self):
        """Create log file if it doesn't exist (empty list)."""
        if not os.path.exists(self.log_file_path):
            self._write_log([])
    
    def _read_log(self) -> List[Dict]:
        """Read existing log entries."""
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _write_log(self, data: List[Dict]):
        """Write log entries to file."""
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def log_qa(self, 
               question: str, 
               answer: str, 
               retrieved_context: List[Dict] = None) -> Dict:
        """
        Log a question-answer pair with retrieved context.
        
        Args:
            question: The user's question
            answer: The assistant's answer (can be JSON string or dict)
            retrieved_context: List of retrieved context chunks with metadata
        
        Returns:
            Dict: The logged entry
        """
        # Parse answer if it's a JSON string
        try:
            if isinstance(answer, str):
                answer_dict = json.loads(answer)
            else:
                answer_dict = answer
        except (json.JSONDecodeError, TypeError):
            answer_dict = answer
        
        # Create the log entry
        entry = {
            "question": question,
            "retrieved_context": retrieved_context or [],
            "answer": answer_dict if isinstance(answer_dict, dict) else {"content": str(answer_dict)}
        }
        
        # Read existing logs
        logs = self._read_log()
        
        # Append new entry
        logs.append(entry)
        
        # Write back to file
        self._write_log(logs)
        
        return entry
    
    def get_all_logs(self) -> List[Dict]:
        """Get all logged Q&A pairs."""
        return self._read_log()
    
    def clear_logs(self):
        """Clear all log entries."""
        self._write_log([])
    
    def get_log_file_path(self) -> str:
        """Get the path to the log file."""
        return self.log_file_path


# Global logger instance
_logger_instance = None

def get_logger(log_file_path: str = None) -> QALogger:
    """Get or create a global QA logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = QALogger(log_file_path)
    return _logger_instance
