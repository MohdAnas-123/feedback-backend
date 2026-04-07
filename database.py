import sqlite3
import threading


class Database:
    """
    Singleton SQLite database wrapper for CritiqueConnect.
    Thread-safe with check_same_thread=False.
    All components share this single instance.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path="critiqueconnect.db"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, db_path="critiqueconnect.db"):
        if self._initialized:
            return
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.connect()
        self.create_tables()
        self._initialized = True

    def connect(self):
        """Connect to the SQLite database with thread-safety enabled."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read performance
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            raise

    def create_tables(self):
        """Create necessary tables if they don't exist."""
        try:
            # Create works table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS works (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create critiques table with expanded columns for paper's Stage 2 analysis
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS critiques (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    work_id INTEGER NOT NULL,
                    aspect TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    cleaned_text TEXT,
                    tone_score REAL,
                    actionability_score REAL,
                    enhanced_text TEXT,
                    sentiment_polarity TEXT,
                    sentiment_intensity REAL,
                    intent TEXT,
                    quality_clarity REAL,
                    quality_specificity REAL,
                    quality_overall REAL,
                    is_meaningful INTEGER DEFAULT 1,
                    cluster_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (work_id) REFERENCES works (id)
                )
            ''')

            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Table creation error: {e}")
            raise

    # ──────────────────────────────────────────────
    #  Works CRUD
    # ──────────────────────────────────────────────

    def add_work(self, user_id, content, work_type):
        """Add a new creative work to the database."""
        try:
            self.cursor.execute(
                "INSERT INTO works (user_id, content, type) VALUES (?, ?, ?)",
                (user_id, content, work_type)
            )
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error adding work: {e}")
            self.conn.rollback()
            return None

    def get_work(self, work_id):
        """Retrieve a work by ID."""
        try:
            self.cursor.execute("SELECT * FROM works WHERE id = ?", (work_id,))
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print(f"Error retrieving work: {e}")
            return None

    # ──────────────────────────────────────────────
    #  Critiques CRUD
    # ──────────────────────────────────────────────

    def add_critique(self, work_id, aspect, raw_text):
        """Add a new raw critique to the database."""
        try:
            self.cursor.execute(
                "INSERT INTO critiques (work_id, aspect, raw_text) VALUES (?, ?, ?)",
                (work_id, aspect, raw_text)
            )
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error adding critique: {e}")
            self.conn.rollback()
            return None

    def get_critiques_for_work(self, work_id):
        """Retrieve all critiques for a specific work."""
        try:
            self.cursor.execute("SELECT * FROM critiques WHERE work_id = ?", (work_id,))
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error retrieving critiques: {e}")
            return []

    # ──────────────────────────────────────────────
    #  Helper methods used by analyzer, enhancer, synthesizer
    # ──────────────────────────────────────────────

    def get_critique_text(self, critique_id):
        """Retrieve the raw text for a specific critique."""
        try:
            self.cursor.execute(
                "SELECT raw_text FROM critiques WHERE id = ?", (critique_id,)
            )
            result = self.cursor.fetchone()
            return result[0] if result else None
        except sqlite3.Error as e:
            print(f"Error retrieving critique text: {e}")
            return None

    def update_critique_preprocessing(self, critique_id, cleaned_text, is_meaningful):
        """Store preprocessing results (Stage 1)."""
        try:
            self.cursor.execute(
                """UPDATE critiques
                   SET cleaned_text = ?, is_meaningful = ?
                   WHERE id = ?""",
                (cleaned_text, int(is_meaningful), critique_id)
            )
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error updating preprocessing: {e}")
            self.conn.rollback()
            return False

    def update_critique_analysis(self, critique_id, tone_score, actionability_score,
                                  sentiment_polarity, sentiment_intensity,
                                  intent, quality_clarity, quality_specificity,
                                  quality_overall):
        """Store full Stage 2 semantic analysis results."""
        try:
            self.cursor.execute(
                """UPDATE critiques
                   SET tone_score = ?, actionability_score = ?,
                       sentiment_polarity = ?, sentiment_intensity = ?,
                       intent = ?,
                       quality_clarity = ?, quality_specificity = ?,
                       quality_overall = ?
                   WHERE id = ?""",
                (tone_score, actionability_score,
                 sentiment_polarity, sentiment_intensity,
                 intent, quality_clarity, quality_specificity,
                 quality_overall, critique_id)
            )
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error updating analysis: {e}")
            self.conn.rollback()
            return False

    def update_critique_cluster(self, critique_id, cluster_id):
        """Store thematic cluster assignment."""
        try:
            self.cursor.execute(
                "UPDATE critiques SET cluster_id = ? WHERE id = ?",
                (cluster_id, critique_id)
            )
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error updating cluster: {e}")
            self.conn.rollback()
            return False

    def update_critique_enhanced_text(self, critique_id, enhanced_text):
        """Store enhanced/synthesized text (Stage 3)."""
        try:
            self.cursor.execute(
                "UPDATE critiques SET enhanced_text = ? WHERE id = ?",
                (enhanced_text, critique_id)
            )
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error updating enhanced text: {e}")
            self.conn.rollback()
            return False

    def get_enhanced_critiques_for_work(self, work_id):
        """Retrieve all enhanced critiques for a work (for synthesis)."""
        try:
            self.cursor.execute(
                """SELECT id, aspect, raw_text, cleaned_text, enhanced_text,
                          tone_score, actionability_score,
                          sentiment_polarity, sentiment_intensity,
                          intent, quality_clarity, quality_specificity,
                          quality_overall, cluster_id
                   FROM critiques
                   WHERE work_id = ? AND is_meaningful = 1""",
                (work_id,)
            )
            columns = [
                "id", "aspect", "raw_text", "cleaned_text", "enhanced_text",
                "tone_score", "actionability_score",
                "sentiment_polarity", "sentiment_intensity",
                "intent", "quality_clarity", "quality_specificity",
                "quality_overall", "cluster_id"
            ]
            rows = self.cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        except sqlite3.Error as e:
            print(f"Error retrieving enhanced critiques: {e}")
            return []

    def get_all_critique_details_for_work(self, work_id):
        """Retrieve full critique details including analysis for report generation."""
        try:
            self.cursor.execute(
                """SELECT id, work_id, aspect, raw_text, cleaned_text,
                          tone_score, actionability_score, enhanced_text,
                          sentiment_polarity, sentiment_intensity,
                          intent, quality_clarity, quality_specificity,
                          quality_overall, is_meaningful, cluster_id, created_at
                   FROM critiques
                   WHERE work_id = ?""",
                (work_id,)
            )
            columns = [
                "id", "work_id", "aspect", "raw_text", "cleaned_text",
                "tone_score", "actionability_score", "enhanced_text",
                "sentiment_polarity", "sentiment_intensity",
                "intent", "quality_clarity", "quality_specificity",
                "quality_overall", "is_meaningful", "cluster_id", "created_at"
            ]
            rows = self.cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        except sqlite3.Error as e:
            print(f"Error retrieving critique details: {e}")
            return []

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            Database._instance = None
            Database._initialized = False