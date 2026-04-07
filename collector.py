from database import Database


class FeedbackCollector:
    """
    Manages collection and storage of creative works and critiques.
    Part of the CritiqueConnect platform's feedback collection layer.
    """

    def __init__(self, db=None):
        """Initialize the FeedbackCollector with a shared database connection."""
        self.db = db or Database()

    def add_work(self, user_id, content, work_type):
        """
        Add a new creative work to the system.

        Parameters:
            user_id (str): Identifier for the creator
            content (str): The creative work content (text format)
            work_type (str): Type of work (e.g., design, writing, code)

        Returns:
            int: ID of the newly created work or None if failed
        """
        if not user_id or not content or not work_type:
            print("Error: All fields (user_id, content, work_type) are required")
            return None

        if not isinstance(content, str):
            print("Error: Work content must be a string")
            return None

        return self.db.add_work(user_id, content, work_type)

    def add_critique(self, work_id, aspect, raw_text):
        """
        Add a new critique to a specific creative work.

        Parameters:
            work_id (int): ID of the work being critiqued
            aspect (str): Aspect being critiqued (e.g., "color scheme", "readability")
            raw_text (str): The raw critique text

        Returns:
            int: ID of the newly created critique or None if failed
        """
        if not work_id or not aspect or not raw_text:
            print("Error: All fields (work_id, aspect, raw_text) are required")
            return None

        if not isinstance(raw_text, str):
            print("Error: Critique text must be a string")
            return None

        # Check if work exists
        work = self.db.get_work(work_id)
        if not work:
            print(f"Error: Work with ID {work_id} not found")
            return None

        return self.db.add_critique(work_id, aspect, raw_text)

    def get_work(self, work_id):
        """
        Retrieve a work by its ID.

        Parameters:
            work_id (int): ID of the work to retrieve

        Returns:
            tuple: Work record or None if not found
        """
        return self.db.get_work(work_id)

    def get_critiques_for_work(self, work_id):
        """
        Retrieve all critiques for a specific work.

        Parameters:
            work_id (int): ID of the work

        Returns:
            list: List of critique records for the given work
        """
        return self.db.get_critiques_for_work(work_id)

    def close(self):
        """Close the database connection."""
        self.db.close()