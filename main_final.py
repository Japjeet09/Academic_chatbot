# app.py
import streamlit as st
import os
from datetime import datetime
import hashlib
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document 
import numpy as np
import tempfile
import cv2
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import easyocr
import speech_recognition as sr
import mysql.connector
import base64
from typing import List, Dict, Optional
import json
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")

class DatabaseManager:
    def __init__(self):
        # Database connection configuration
        self.db_config = {
            "pool_name": "mypool",
            "pool_size": 5,
            "host": os.getenv("MYSQL_HOST"),
            "user": os.getenv("MYSQL_USER"),
            "password": os.getenv("MYSQL_PASSWORD"),
            "database": os.getenv("MYSQL_DATABASE")
        }
        
        # Initialize connection pool
        try:
            self.connection_pool = mysql.connector.pooling.MySQLConnectionPool(**self.db_config)
        except mysql.connector.Error as err:
            print(f"Error connecting to MySQL: {err}")
            raise
            
    def initialize_database(self):
        """Create necessary tables if they don't exist"""
        connection = self.connection_pool.get_connection()
        cursor = connection.cursor()
        
        try:
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Subjects table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS subjects (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    subject_name VARCHAR(255) NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_modified DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    UNIQUE KEY unique_user_subject (user_id, subject_name)
                )
            """)
            
            # Topics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS topics (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    subject_id INT NOT NULL,
                    topics_name TEXT NOT NULL,
                    is_custom BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (subject_id) REFERENCES subjects(id)
                )
            """)
            
            # Chat history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    subject_id INT,
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    FOREIGN KEY (subject_id) REFERENCES subjects(id)
                )
            """)
            
            connection.commit()
            
        except mysql.connector.Error as err:
            print(f"Error creating tables: {err}")
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()
    
    def add_user(self, username: str, hashed_password: str) -> Optional[int]:
        """Add a new user and return their ID"""
        connection = self.connection_pool.get_connection()
        cursor = connection.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (%s, %s)",
                (username, hashed_password)
            )
            connection.commit()
            return cursor.lastrowid
        except mysql.connector.Error:
            connection.rollback()
            return None
        finally:
            cursor.close()
            connection.close()
    
    def get_user_id(self, username: str) -> Optional[int]:
        """Get user ID from username"""
        connection = self.connection_pool.get_connection()
        cursor = connection.cursor()
        
        try:
            cursor.execute(
                "SELECT id FROM users WHERE username = %s",
                (username,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
        finally:
            cursor.close()
            connection.close()
            
    def verify_user(self, username: str, hashed_password: str) -> bool:
        """Verify user credentials"""
        connection = self.connection_pool.get_connection()
        cursor = connection.cursor()
        
        try:
            cursor.execute(
                "SELECT id FROM users WHERE username = %s AND password = %s",
                (username, hashed_password)
            )
            return cursor.fetchone() is not None
        finally:
            cursor.close()
            connection.close()
    
    def add_subject(self, user_id: int, subject_name: str) -> Optional[int]:
        """Add a new subject for a user"""
        connection = self.connection_pool.get_connection()
        cursor = connection.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO subjects (user_id, subject_name) VALUES (%s, %s)",
                (user_id, subject_name)
            )
            connection.commit()
            return cursor.lastrowid
        except mysql.connector.Error:
            connection.rollback()
            return None
        finally:
            cursor.close()
            connection.close()
    
    def get_user_subjects(self, user_id: int) -> List[Dict]:
        """Get all subjects for a user"""
        connection = self.connection_pool.get_connection()
        cursor = connection.cursor(dictionary=True)
        
        try:
            cursor.execute(
                "SELECT * FROM subjects WHERE user_id = %s",
                (user_id,)
            )
            return cursor.fetchall()
        finally:
            cursor.close()
            connection.close()
    
    def clear_syllabus_topics(self, subject_id: int) -> bool:
        """
        Clear non-custom topics for a specific subject
        
        Args:
            subject_id (int): ID of the subject to clear topics for
        
        Returns:
            bool: True if topics were cleared successfully, False otherwise
        """
        connection = self.connection_pool.get_connection()
        cursor = connection.cursor()
        
        try:
            # Delete only non-custom topics
            delete_query = """
            DELETE FROM topics 
            WHERE subject_id = %s AND is_custom = FALSE
            """
            cursor.execute(delete_query, (subject_id,))
            connection.commit()
            return True
        except mysql.connector.Error as e:
            print(f"Error clearing syllabus topics: {e}")
            connection.rollback()
            return False
        finally:
            cursor.close()
            connection.close()
        
    def add_topic(self, subject_id: int, topics_name: str, is_custom: bool = False) -> Optional[int]:
        """Add a topic to a subject using correct column name"""
        connection = self.connection_pool.get_connection()
        cursor = connection.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO topics (subject_id, topics_name, is_custom) VALUES (%s, %s, %s)",
                (subject_id, topics_name, is_custom)
            )
            connection.commit()
            return cursor.lastrowid
        except mysql.connector.Error as e:
            print(f"Error adding topic: {e}")
            connection.rollback()
            return None
        finally:
            cursor.close()
            connection.close()
    
    def get_subject_topics(self, subject_id: int) -> Dict[str, List[str]]:
        """Get all topics for a subject using correct column name"""
        connection = self.connection_pool.get_connection()
        cursor = connection.cursor(dictionary=True)
        
        try:
            cursor.execute(
                "SELECT topics_name, is_custom FROM topics WHERE subject_id = %s",
                (subject_id,)
            )
            results = cursor.fetchall()
            
            syllabus_topics = []
            custom_topics = []
            
            for row in results:
                if row['is_custom']:
                    custom_topics.append(row['topics_name'])
                else:
                    syllabus_topics.append(row['topics_name'])
                    
            return {
                'syllabus_topics': syllabus_topics,
                'custom_topics': custom_topics
            }
        finally:
            cursor.close()
            connection.close()
    
    def add_chat_message(self, user_id: int, subject_id: Optional[int], role: str, content: str) -> Optional[int]:
        """Add a chat message to history"""
        connection = self.connection_pool.get_connection()
        cursor = connection.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO chat_history (user_id, subject_id, role, content) VALUES (%s, %s, %s, %s)",
                (user_id, subject_id, role, content)
            )
            connection.commit()
            return cursor.lastrowid
        except mysql.connector.Error:
            connection.rollback()
            return None
        finally:
            cursor.close()
            connection.close()
    
    def get_chat_history(self, user_id: int, subject_id: Optional[int] = None, limit: int = 50) -> List[Dict]:
        """Get chat history for a user and optionally for a specific subject"""
        connection = self.connection_pool.get_connection()
        cursor = connection.cursor(dictionary=True)
        
        try:
            if subject_id:
                cursor.execute(
                    """
                    SELECT role, content, created_at 
                    FROM chat_history 
                    WHERE user_id = %s AND subject_id = %s 
                    ORDER BY created_at DESC 
                    LIMIT %s
                    """,
                    (user_id, subject_id, limit)
                )
            else:
                cursor.execute(
                    """
                    SELECT role, content, created_at 
                    FROM chat_history 
                    WHERE user_id = %s 
                    ORDER BY created_at DESC 
                    LIMIT %s
                    """,
                    (user_id, limit)
                )
            
            return cursor.fetchall()
        finally:
            cursor.close()
            connection.close()
    
    def delete_subject(self, subject_id: int) -> bool:
        """Delete a subject and all associated data"""
        connection = self.connection_pool.get_connection()
        cursor = connection.cursor()
        
        try:
            # Delete associated chat history
            cursor.execute(
                "DELETE FROM chat_history WHERE subject_id = %s",
                (subject_id,)
            )
            
            # Delete associated topics
            cursor.execute(
                "DELETE FROM topics WHERE subject_id = %s",
                (subject_id,)
            )
            
            # Delete the subject
            cursor.execute(
                "DELETE FROM subjects WHERE id = %s",
                (subject_id,)
            )
            
            connection.commit()
            return True
        except mysql.connector.Error:
            connection.rollback()
            return False
        finally:
            cursor.close()
            connection.close()

    def delete_topic(self, topic_id: int) -> bool:
        """
        Delete a topic by its ID with corrected column names
        
        Args:
            topic_id (int): The ID of the topic to delete
            
        Returns:
            bool: True if topic was successfully deleted, False otherwise
        """
        connection = None
        cursor = None
        
        try:
            # Validate topic_id
            if not isinstance(topic_id, int):
                raise ValueError(f"Topic ID must be an integer, got {type(topic_id)}")
                
            if topic_id <= 0:
                raise ValueError(f"Invalid topic ID: {topic_id}")
                
            connection = self.connection_pool.get_connection()
            cursor = connection.cursor()
            
            # First verify the topic exists - using correct column name 'topics_name'
            cursor.execute(
                "SELECT id, topics_name FROM topics WHERE id = %s",
                (topic_id,)
            )
            topic = cursor.fetchone()
            
            if not topic:
                print(f"Topic with ID {topic_id} not found")
                return False
                
            # Proceed with deletion
            cursor.execute(
                "DELETE FROM topics WHERE id = %s",
                (topic_id,)
            )
            connection.commit()
            
            affected_rows = cursor.rowcount
            if affected_rows > 0:
                print(f"Successfully deleted topic with ID {topic_id}")
                return True
            else:
                print(f"No topic was deleted for ID {topic_id}")
                return False
                
        except ValueError as e:
            print(f"Validation error: {str(e)}")
            if connection:
                connection.rollback()
            return False
            
        except mysql.connector.Error as e:
            print(f"Database error while deleting topic: {str(e)}")
            if connection:
                connection.rollback()
            return False
            
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

class VoiceChatManager:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
    def listen_for_speech(self) -> str:
        """Record and transcribe speech to text"""
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                st.info("Listening... Speak now!")
                
                # Record audio
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
                
                st.info("Processing speech...")
                
                # Convert speech to text
                text = self.recognizer.recognize_google(audio)
                return text
                
        except sr.WaitTimeoutError:
            st.warning("No speech detected. Please try again.")
            return ""
        except sr.UnknownValueError:
            st.warning("Could not understand audio. Please try again.")
            return ""
        except sr.RequestError as e:
            st.error(f"Could not request results; {str(e)}")
            return ""
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return ""
        

def add_voice_chat_ui(tab1):
    """Add voice chat UI elements to the chat tab"""
    
    # Initialize voice chat manager in session state if not exists
    if 'voice_manager' not in st.session_state:
        st.session_state.voice_manager = VoiceChatManager()
    
    # Voice interaction control
    if st.button("🎤 Start Voice Input", key="voice_input"):
        text = st.session_state.voice_manager.listen_for_speech()
        if text:
            # Update chat with transcribed text
            with st.chat_message("user"):
                st.write(text)
            
            # Get chatbot response
            with st.chat_message("assistant"):
                response = st.session_state.chatbot.get_response(
                    text,
                    st.session_state.user,
                    st.session_state.current_subject,
                    st.session_state.subject_chat_histories,
                    st.session_state.use_syllabus_context
                )
                st.write(response)
            
            # Add to chat history
            current_history = st.session_state.subject_chat_histories.get(
                st.session_state.current_subject, 
                []
            )
            current_history.extend([
                {"role": "user", "content": text},
                {"role": "assistant", "content": response}
            ])
            st.session_state.subject_chat_histories[st.session_state.current_subject] = current_history

# Function to display the PDF of a given file
def displayPDF(file):
    # Reading the uploaded file
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying the PDF
    st.markdown(pdf_display, unsafe_allow_html=True)

class SyllabusManager:
    def __init__(self):
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.client = QdrantClient(url=qdrant_url, prefer_grpc=False)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=250
        )
        self.reader = easyocr.Reader(['en'], gpu=False)
        
        self.pdf_params = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            boxes_flow=0.5
        )

    def display_file_preview(self, file, st) -> None:
        """Display a preview of the uploaded file"""
        try:
            file_extension = file.name.split('.')[-1].lower()

            if 'show_file_preview' not in st.session_state:
                st.session_state.show_file_preview = False

            # Create a button to toggle the preview visibility
            if st.button("📄 Show/Hide Syllabus"):
                st.session_state.show_file_preview = not st.session_state.show_file_preview
            
            if st.session_state.show_file_preview:
                if file_extension in ['png', 'jpg', 'jpeg']:
                    # Create columns for image display and info
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.image(file, caption="Uploaded Image", use_column_width=True)
                    with col2:
                        st.write("File Information:")
                        st.write(f"- Name: {file.name}")
                        st.write(f"- Size: {file.size//1024} KB")
                        st.write(f"- Type: {file.type}")
            
            elif file_extension == 'pdf':
                # Display PDF information and first page preview
                st.write("PDF File Information:")
                # st.write(f"- Name: {file.name}")
                # st.write(f"- Size: {file.size//1024} KB")
                
                # # Create a temporary file to read PDF content
                # with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                #     tmp_file.write(file.read())
                #     tmp_path = tmp_file.name
                
                # # Extract first page text for preview
                # extracted_text = extract_text(tmp_path, maxpages=1, laparams=self.pdf_params)
                # with st.expander("Preview PDF Content (First Page)"):
                #     st.text(extracted_text[:1000] + "...")
                # Display file name and size
                st.markdown(f"**Filename:** {file.name}")
                st.markdown(f"**File Size:** {file.size} bytes")
                
                # Display PDF preview using displayPDF function
                st.markdown("### 📖 PDF Preview")
                displayPDF(file)
                
                # Save the uploaded file to a temporary location
                tmp_path = "temp.pdf"
                with open(tmp_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Store the temp_pdf_path in session_state
                st.session_state['temp_pdf_path'] = tmp_path
                
                # Reset file pointer for later processing
                file.seek(0)
                os.unlink(tmp_path)
                
        except Exception as e:
            st.error(f"Error displaying preview: {str(e)}")

    def _process_image(self, image_content: bytes, username: str, subject_name: str = None) -> List[str]:
        """Enhanced image processing with improved OCR accuracy"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Enhanced preprocessing pipeline
            processed_image = self._enhanced_preprocess_image(image)
            
            # Perform OCR with improved settings
            results = self._perform_enhanced_ocr(processed_image)
            
            # Post-process OCR results
            text = self._post_process_ocr_results(results)
            
            # Create document
            documents = [Document(
                page_content=text,
                metadata={
                    "source": "image",
                    "type": "syllabus",
                    "subject": subject_name,
                    "processed_date": datetime.now().isoformat()
                }
            )]
            
            # Split text
            texts = self.text_splitter.split_documents(documents)
            
            # Store in Qdrant
            collection_name = f"syllabus_{username}"
            if subject_name:
                collection_name += f"_{subject_name}"
                
            Qdrant.from_documents(
                documents=texts,
                embedding=self.embeddings,
                url=qdrant_url,
                prefer_grpc=False,
                collection_name=collection_name
            )

            # Extract topics
            topics = self._extract_topics_enhanced(texts)
            return topics
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return []

    def _enhanced_preprocess_image(self, image):
        """Enhanced image preprocessing pipeline"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        # Apply dilation to connect components
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(denoised, kernel, iterations=1)
        
        # Apply erosion to remove noise
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(eroded)
        
        return enhanced

    def _perform_enhanced_ocr(self, image):
        """Perform OCR with enhanced settings"""
        # Perform OCR with better settings
        results = self.reader.readtext(
            image,
            paragraph=True,
            batch_size=8,
            min_size=10,
            contrast_ths=0.1,
            adjust_contrast=0.8,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            mag_ratio=1.5
        )
        
        return results

    def _post_process_ocr_results(self, results):
        """Post-process OCR results for better accuracy"""
        processed_lines = []
        current_paragraph = []
        
        for result in results:
            if isinstance(result, str):
                text = result
            else:
                text = result[1] if len(result) > 1 else ""
            
            # Clean up the text
            text = text.strip()
            if not text:
                continue
            
            # Handle line breaks and paragraphs
            if len(text) > 50 or text.endswith(('.', ':', ')')):
                if current_paragraph:
                    current_paragraph.append(text)
                    processed_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                else:
                    processed_lines.append(text)
            else:
                current_paragraph.append(text)
        
        # Add any remaining paragraph
        if current_paragraph:
            processed_lines.append(' '.join(current_paragraph))
        
        return '\n'.join(processed_lines)

    def process_file(self, file, username: str, subject_name: str = None) -> List[str]:
        """Process uploaded file with preview functionality"""
        try:
            # Display file preview
            self.display_file_preview(file, st)
            
            file_content = file.read()
            file_extension = file.name.split('.')[-1].lower()

            if subject_name:
                # Create directory if it doesn't exist
                os.makedirs(f"user_files/{username}", exist_ok=True)

            # Save the file with a standardized name
            file_path = f"user_files/{username}/{subject_name}.{file_extension}"
            with open(file_path, "wb") as f:
                f.write(file_content)

            # Store the file path in metadata
            file_metadata = {
                "file_path": file_path,
                "file_type": file_extension,
                "upload_date": datetime.now().isoformat()
            }

            # Save metadata to a JSON file or database
            metadata_path = f"user_files/{username}/{subject_name}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(file_metadata, f)

            # Reset file pointer for processing
            file.seek(0)
            
            if file_extension == 'pdf':
                return self._process_pdf(file_content, username, subject_name)
            elif file_extension in ['png', 'jpg', 'jpeg']:
                return self._process_image(file_content, username, subject_name)
            else:
                st.error("Unsupported file format")
                return []
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return []

    def _process_pdf(self, pdf_content: bytes, username: str, subject_name: str = None) -> List[str]:
        """Process PDF with optional subject specification"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_content)
            tmp_path = tmp_file.name

        try:
            extracted_text = extract_text(tmp_path, laparams=self.pdf_params)
            documents = [Document(
                page_content=extracted_text,
                metadata={"source": "pdf", "type": "syllabus", "subject": subject_name}
            )]
            
            texts = self.text_splitter.split_documents(documents)
            
            collection_name = f"syllabus_{username}"
            if subject_name:
                collection_name += f"_{subject_name}"
            
            Qdrant.from_documents(
                documents=texts,
                embedding=self.embeddings,
                url=qdrant_url,
                prefer_grpc=False,
                collection_name=collection_name
            )

            topics = self._extract_topics_enhanced(texts)
            return topics
        finally:
            os.unlink(tmp_path)

    def display_subject_syllabus(self, username: str, subject_name: str, st) -> None:
        """Display the syllabus file for a previously uploaded subject"""
        try:
            # Check if metadata exists for this subject
            metadata_path = f"user_files/{username}/{subject_name}_metadata.json"
            if not os.path.exists(metadata_path):
                st.warning(f"No syllabus file found for {subject_name}")
                return
                
            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                
            # Get file path and type
            file_path = metadata["file_path"]
            file_type = metadata["file_type"]
            
            # Initialize toggle state if needed
            if 'show_subject_syllabus' not in st.session_state:
                st.session_state.show_subject_syllabus = False
                
            # Create toggle button with dynamic text
            button_text = "Hide Syllabus" if st.session_state.show_subject_syllabus else "Show Syllabus"
            if st.button(f"📄 {button_text}"):
                st.session_state.show_subject_syllabus = not st.session_state.show_subject_syllabus
                
            # Display syllabus if toggled on
            if st.session_state.show_subject_syllabus:
                st.markdown(f"### 📚 Syllabus for {subject_name}")
                
                if file_type == 'pdf':
                    # Open the stored PDF file
                    with open(file_path, "rb") as f:
                        pdf_data = f.read()
                        
                    # Display PDF
                    st.markdown(f"**File Type:** PDF")
                    displayPDF(BytesIO(pdf_data))
                    
                elif file_type in ['png', 'jpg', 'jpeg']:
                    # Display image
                    st.markdown(f"**File Type:** Image")
                    st.image(file_path, caption=f"{subject_name} Syllabus", use_column_width=True)
                    
        except Exception as e:
            st.error(f"Error displaying syllabus: {str(e)}")
        
    def delete_subject_collection(self, username: str, subject_name: str) -> bool:
        """Delete a subject's collection from Qdrant"""
        try:
            collection_name = f"syllabus_{username}_{subject_name}"
            self.client.delete_collection(collection_name=collection_name)
            return True
        except Exception as e:
            st.error(f"Error deleting collection from vector database: {str(e)}")
            return False

    def _extract_topics_enhanced(self, texts) -> List[str]:
        """Enhanced topic extraction with better pattern recognition"""
        topics = []
        common_unit_patterns = [
            r"Unit[- ]\d+",
            r"Chapter[- ]\d+",
            r"Module[- ]\d+",
            r"Section[- ]\d+"
        ]
        
        for text in texts:
            content = text.page_content
            lines = content.split('\n')
            current_unit = None
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and very long lines
                if not line or len(line) > 200:
                    continue
                
                # Check for unit headers
                import re
                for pattern in common_unit_patterns:
                    if re.match(pattern, line, re.IGNORECASE):
                        current_unit = line
                        topics.append(line)
                        break
                
                # Check for topic lines
                if (len(line) > 3 and len(line) < 150 and
                    (any(keyword in line.lower() for keyword in [
                        'introduction', 'overview', 'fundamental', 'basic',
                        'advanced', 'application', 'method', 'technique',
                        'analysis', 'processing', 'system', 'model'
                    ]) or
                    (line[0].isupper() and ':' in line) or
                    (current_unit and line.strip().startswith('-')))):
                    
                    # Clean up the topic
                    topic = line.strip('- :.')
                    if current_unit:
                        topic = f"{current_unit}: {topic}"
                    topics.append(topic)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_topics = []
        for topic in topics:
            if topic.lower() not in seen:
                seen.add(topic.lower())
                unique_topics.append(topic)
        
        return unique_topics

    def get_relevant_context(self, query: str, username: str, subject_name: str = None) -> str:
        """Get relevant context with optional subject specification"""
        try:
            collection_name = f"syllabus_{username}"
            if subject_name:
                collection_name += f"_{subject_name}"
                
            vectorstore = Qdrant(
                client=self.client,
                embeddings=self.embeddings,
                collection_name=collection_name
            )
            
            docs = vectorstore.max_marginal_relevance_search(
                query,
                k=5,
                fetch_k=8,
                lambda_mult=0.7
            )
            
            contexts = []
            for i, doc in enumerate(docs, 1):
                context = f"Section {i}:\n{doc.page_content.strip()}"
                contexts.append(context)
            
            return "\n\n".join(contexts)
        except Exception as e:
            return ""

class ExamMode:
    def __init__(self, llm):
        self.llm = llm
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.client = QdrantClient(url=qdrant_url, prefer_grpc=False)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=250
        )
        self.pdf_params = LAParams()
        self.reader = easyocr.Reader(['en'])

        self.probable_questions_template = """
        Based on the analysis of previous year questions and patterns, generate probable questions for the upcoming exam.
        
        Previous Questions and Patterns:
        {context}
        
        Subject: {subject}
        
        Generate {num_questions} probable questions that might appear in the exam. Include a mix of:
        - Theory questions (definitions, explanations, comparisons, case studies)
        - Numerical problems (calculations, derivations, problem-solving)
        
        For each question:
        1. Consider frequently tested topics
        2. Follow common question patterns from previous years
        3. Match the difficulty level distribution
        4. Provide detailed solutions/answers
        
        Format each question as follows:
        Question Number: [number]
        Type: [Theory/Numerical]
        Marks: [expected marks]
        Difficulty: [Easy/Medium/Hard]
        Question: [The complete question text]
        Solution: [Detailed step-by-step solution or answer points]
        Key Points to Remember: [Important concepts, formulas, or approaches]
        Previous Year Pattern: [Mention if similar questions appeared before and how]
        Important Topics Covered: [List main topics this question covers]
        Examiner's Notes: [What the examiner typically looks for in the answer]
        
        END_OF_QUESTION
        """
        
        self.quiz_template = """
        Based on the context provided and the topic, create a challenging quiz question.
        Context: {context}
        Topic: {topic}
        
        Format your response EXACTLY as follows:
        Question: [Your question here]
        A) [Option]
        B) [Option]
        C) [Option]
        D) [Option]
        Correct Answer: [Single letter A, B, C, or D]
        Explanation: [Detailed explanation]
        Level: [Easy/Medium/Hard]
        Category: [Theory/Numerical/Conceptual]
        """
        
        self.analysis_template = """
        Analyze the following previous year questions and extract key topics and patterns:
        Questions: {questions}
        
        Provide your analysis in the following format:
        Common Topics:
        - [List of recurring topics]
        Question Patterns:
        - [List of common question patterns]
        Important Concepts:
        - [List of key concepts to focus on]
        Recommended Focus Areas:
        - [Areas that frequently appear in exams]
        """
    
    def process_previous_papers(self, username: str, subject: str, uploaded_files) -> Dict:
        """Process and store previous year question papers with comprehensive error handling"""
        all_documents = []
        processing_results = {
            "success": False,
            "processed_files": [],
            "failed_files": [],
            "total_documents": 0,
            "error": None
        }
        
        try:
            # Process each uploaded file
            for uploaded_file in uploaded_files:
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                tmp_path = None
                
                try:
                    if file_extension == '.pdf':
                        # Process PDF file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name
                        
                        extracted_text = extract_text(tmp_path, laparams=self.pdf_params)
                        if not extracted_text.strip():
                            raise ValueError("No text could be extracted from PDF")
                            
                        all_documents.append(Document(
                            page_content=extracted_text,
                            metadata={
                                "source": "previous_paper",
                                "type": "exam",
                                "subject": subject,
                                "filename": uploaded_file.name
                            }
                        ))
                        processing_results["processed_files"].append(uploaded_file.name)
                        
                    elif file_extension in ['.png', '.jpg', '.jpeg']:
                        # Process image file
                        try:
                            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
                            results = self.reader.readtext(image)
                            
                            if not results:
                                raise ValueError("No text could be extracted from image")
                                
                            text = ' '.join([result[1] for result in results])
                            all_documents.append(Document(
                                page_content=text,
                                metadata={
                                    "source": "previous_paper",
                                    "type": "exam",
                                    "subject": subject,
                                    "filename": uploaded_file.name
                                }
                            ))
                            processing_results["processed_files"].append(uploaded_file.name)
                            
                        except Exception as e:
                            processing_results["failed_files"].append({
                                "filename": uploaded_file.name,
                                "error": f"Image processing failed: {str(e)}"
                            })
                            
                    else:
                        processing_results["failed_files"].append({
                            "filename": uploaded_file.name,
                            "error": "Unsupported file format"
                        })
                        
                except Exception as e:
                    processing_results["failed_files"].append({
                        "filename": uploaded_file.name,
                        "error": str(e)
                    })
                    
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
            
            if not all_documents:
                raise ValueError("No documents were successfully processed")
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(all_documents)
            
            # Store in Qdrant with collection handling
            collection_name = f"previous_papers_{username}_{subject}"
            try:
                # Check if collection exists
                try:
                    self.client.get_collection(collection_name)
                except Exception:
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            "size": self.embeddings.model_kwargs.get("dimension", 384),
                            "distance": "Cosine"
                        }
                    )
                
                # Store documents
                Qdrant.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    url=qdrant_url,
                    prefer_grpc=False,
                    collection_name=collection_name
                )
                
                # Analyze patterns if any documents were processed
                prompt = PromptTemplate(
                    template=self.analysis_template,
                    input_variables=["questions"]
                )
                
                combined_text = "\n".join([doc.page_content for doc in all_documents])
                analysis = self.llm.predict(prompt.format(questions=combined_text[:5000]))
                
                processing_results.update({
                    "success": True,
                    "total_documents": len(chunks),
                    "analysis": analysis
                })
                
            except Exception as e:
                raise ValueError(f"Failed to store in vector database: {str(e)}")
                
            return processing_results
            
        except Exception as e:
            processing_results["error"] = str(e)
            return processing_results
    
    def generate_probable_questions(self, username: str, subject_name: str, num_questions: int = 10) -> List[Dict]:
        """
        Generate probable theory and numerical questions for upcoming exams based on previous year papers analysis.
        
        Args:
            username (str): The username
            subject_name (str): The subject name
            num_questions (int): Number of probable questions to generate
            
        Returns:
            List[Dict]: List of probable questions with detailed information
        """
        try:
            # Get context from previous papers
            collection_name = f"previous_papers_{username}_{subject_name}"
            vectorstore = Qdrant(
                client=self.client,
                embeddings=self.embeddings,
                collection_name=collection_name
            )
            
            # Get all relevant documents for analysis
            context_docs = vectorstore.similarity_search("exam questions patterns", k=10)
            context = "\n".join([doc.page_content for doc in context_docs])
            
            # Generate probable questions
            prompt = PromptTemplate(
                template=self.probable_questions_template,
                input_variables=["context", "subject", "num_questions"]
            )
            
            response = self.llm.predict(prompt.format(
                context=context,
                subject=subject_name,
                num_questions=num_questions
            ))
            
            # Parse the response into structured format
            questions = []
            current_question = {}
            current_field = None
            
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if line == "END_OF_QUESTION":
                    if current_question:
                        questions.append(current_question)
                        current_question = {}
                        current_field = None
                    continue
                
                # Handle field identification
                field_mappings = {
                    "Question Number:": "question_number",
                    "Type:": "type",
                    "Marks:": "marks",
                    "Difficulty:": "difficulty",
                    "Question:": "question_text",
                    "Solution:": "solution",
                    "Key Points to Remember:": "key_points",
                    "Previous Year Pattern:": "previous_pattern",
                    "Important Topics Covered:": "topics_covered",
                    "Examiner's Notes:": "examiner_notes"
                }
                
                # Check if line starts with a new field
                for prefix, field_name in field_mappings.items():
                    if line.startswith(prefix):
                        current_field = field_name
                        content = line[len(prefix):].strip()
                        current_question[field_name] = content
                        break
                else:
                    # If no new field is found, append to current field
                    if current_field and current_question:
                        current_question[current_field] += "\n" + line
            
            # Add the last question if exists
            if current_question:
                questions.append(current_question)
            
            return questions
            
        except Exception as e:
            print(f"Error generating probable questions: {str(e)}")
            return []

    def analyze_question_patterns(self, username: str, subject_name: str) -> Dict:
        """
        Analyze patterns in previous year questions to guide probable question generation.
        
        Returns:
            Dict containing analysis of question patterns, mark distribution, and topic frequency
        """
        try:
            collection_name = f"previous_papers_{username}_{subject_name}"
            vectorstore = Qdrant(
                client=self.client,
                embeddings=self.embeddings,
                collection_name=collection_name
            )
            
            # Get all documents for analysis
            docs = vectorstore.similarity_search("exam questions", k=100)
            
            # Custom prompt for detailed pattern analysis
            analysis_prompt = """
            Analyze these exam questions and provide:
            1. Common question patterns for theory questions
            2. Common types of numerical problems
            3. Mark distribution patterns
            4. Frequently tested topics
            5. Common question structures
            
            Questions:
            {questions}
            """
            
            prompt = PromptTemplate(
                template=analysis_prompt,
                input_variables=["questions"]
            )
            
            analysis = self.llm.predict(prompt.format(
                questions="\n".join([doc.page_content for doc in docs])
            ))
            
            return {
                "raw_analysis": analysis,
                "document_count": len(docs),
                "collection_name": collection_name
            }
            
        except Exception as e:
            print(f"Error analyzing question patterns: {str(e)}")
            return {}
            
    def get_topic_wise_probable_questions(self, username: str, subject_name: str, 
                                        topics: List[str], questions_per_topic: int = 3) -> Dict[str, List[Dict]]:
        """
        Generate probable questions organized by topics.
        
        Args:
            username (str): The username
            subject_name (str): The subject name
            topics (List[str]): List of topics to generate questions for
            questions_per_topic (int): Number of questions to generate per topic
            
        Returns:
            Dict[str, List[Dict]]: Dictionary mapping topics to their probable questions
        """
        topic_questions = {}
        
        for topic in topics:
            questions = self.generate_probable_questions(
                username=username,
                subject_name=subject_name,
                num_questions=questions_per_topic
            )
            topic_questions[topic] = questions
            
        return topic_questions
    
    def generate_adaptive_quiz(self, topic: str, username: str, subject_name: str, 
                             difficulty: str = "Medium") -> Dict:
        """Generate quiz questions based on previous papers and user performance"""
        try:
            # Get relevant context from previous papers
            collection_name = f"previous_papers_{username}_{subject_name}"
            vectorstore = Qdrant(
                client=self.client,
                embeddings=self.embeddings,
                collection_name=collection_name
            )
            
            context_docs = vectorstore.similarity_search(topic, k=3)
            context = "\n".join([doc.page_content for doc in context_docs])
            
            prompt = PromptTemplate(
                template=self.quiz_template,
                input_variables=["context", "topic"]
            )
            
            response = self.llm.predict(prompt.format(
                context=context,
                topic=topic
            ))
            
            # Parse response into structured format
            lines = response.split('\n')
            quiz_data = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('Question:'):
                    quiz_data['question'] = line.replace('Question:', '').strip()
                elif line.startswith(('A)', 'B)', 'C)', 'D)')):
                    if 'options' not in quiz_data:
                        quiz_data['options'] = []
                    quiz_data['options'].append(line.strip())
                elif line.startswith('Correct Answer:'):
                    quiz_data['correct'] = line.replace('Correct Answer:', '').strip()[0].upper()
                elif line.startswith('Explanation:'):
                    quiz_data['explanation'] = line.replace('Explanation:', '').strip()
                elif line.startswith('Level:'):
                    quiz_data['level'] = line.replace('Level:', '').strip()
                elif line.startswith('Category:'):
                    quiz_data['category'] = line.replace('Category:', '').strip()
            
            return quiz_data
            
        except Exception as e:
            return None
    
    def track_performance(self, username: str, subject_name: str, topic: str,
                         question_data: Dict, user_answer: str, is_correct: bool) -> None:
        """Track user performance for adaptive learning"""
        collection_name = f"performance_{username}_{subject_name}"
        
        performance_data = {
            "topic": topic,
            "question": question_data['question'],
            "category": question_data.get('category', 'Unknown'),
            "level": question_data.get('level', 'Medium'),
            "is_correct": is_correct,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store performance data in vector database
        doc = Document(
            page_content=str(performance_data),
            metadata=performance_data
        )
        
        Qdrant.from_documents(
            documents=[doc],
            embedding=self.embeddings,
            url=qdrant_url,
            prefer_grpc=False,
            collection_name=collection_name
        )
    
    def get_weak_areas(self, username: str, subject_name: str) -> List[Dict]:
        """Analyze user performance to identify weak areas with detailed topic and question analysis"""
        try:
            collection_name = f"performance_{username}_{subject_name}"
            vectorstore = Qdrant(
                client=self.client,
                embeddings=self.embeddings,
                collection_name=collection_name
            )
            
            # Get all performance records
            results = vectorstore.similarity_search("performance analysis", k=100)
            
            # Analyze performance by topic and category
            topic_stats = {}
            
            for result in results:
                metadata = result.metadata
                topic = metadata['topic']
                
                if topic not in topic_stats:
                    topic_stats[topic] = {
                        'total': 0,
                        'correct': 0,
                        'categories': set(),
                        'recent_performance': [],
                        'questions': [],  # Track individual questions
                        'subtopics': {},  # Track performance by subtopic
                        'difficulty_levels': {'Easy': 0, 'Medium': 0, 'Hard': 0},
                        'common_mistakes': []  # Track patterns in incorrect answers
                    }
                
                stats = topic_stats[topic]
                stats['total'] += 1
                if metadata['is_correct']:
                    stats['correct'] += 1
                
                stats['categories'].add(metadata['category'])
                stats['recent_performance'].append(metadata['is_correct'])
                
                # Track question details
                question_detail = {
                    'question': metadata.get('question', 'Unknown question'),
                    'correct': metadata['is_correct'],
                    'timestamp': metadata.get('timestamp', ''),
                    'level': metadata.get('level', 'Medium'),
                    'category': metadata['category']
                }
                stats['questions'].append(question_detail)
                
                # Update difficulty level counts
                level = metadata.get('level', 'Medium')
                stats['difficulty_levels'][level] += 1
                
                # Track subtopic performance if available
                subtopic = metadata.get('subtopic', 'General')
                if subtopic not in stats['subtopics']:
                    stats['subtopics'][subtopic] = {'total': 0, 'correct': 0}
                stats['subtopics'][subtopic]['total'] += 1
                if metadata['is_correct']:
                    stats['subtopics'][subtopic]['correct'] += 1
            
            # Calculate weak areas with detailed analysis
            weak_areas = []
            for topic, stats in topic_stats.items():
                accuracy = (stats['correct'] / stats['total']) * 100
                recent_trend = np.mean(stats['recent_performance'][-5:]) * 100 if len(stats['recent_performance']) >= 5 else accuracy
                
                # Calculate subtopic performance
                weak_subtopics = []
                for subtopic, sub_stats in stats['subtopics'].items():
                    if sub_stats['total'] > 0:
                        sub_accuracy = (sub_stats['correct'] / sub_stats['total']) * 100
                        if sub_accuracy < 70:
                            weak_subtopics.append({
                                'name': subtopic,
                                'accuracy': sub_accuracy,
                                'attempts': sub_stats['total']
                            })
                
                # Analyze recent questions
                recent_questions = sorted(stats['questions'], 
                                    key=lambda x: x['timestamp'], 
                                    reverse=True)[:5]
                
                # Calculate difficulty distribution
                total_questions = sum(stats['difficulty_levels'].values())
                difficulty_distribution = {
                    level: (count / total_questions * 100) 
                    for level, count in stats['difficulty_levels'].items()
                }
                
                if accuracy < 70 or recent_trend < 60:
                    weak_areas.append({
                        'topic': topic,
                        'accuracy': accuracy,
                        'recent_trend': recent_trend,
                        'categories': list(stats['categories']),
                        'total_questions': stats['total'],
                        'weak_subtopics': weak_subtopics,
                        'recent_questions': recent_questions,
                        'difficulty_distribution': difficulty_distribution,
                        'performance_by_difficulty': {
                            level: {
                                'count': count,
                                'percentage': (count / total_questions * 100)
                            }
                            for level, count in stats['difficulty_levels'].items()
                        }
                    })
                
            return sorted(weak_areas, key=lambda x: x['accuracy'])
            
        except Exception as e:
            print(f"Error analyzing weak areas: {str(e)}")
            return []

class UserManager:
    def __init__(self):
        self.db = DatabaseManager()
        self.db.initialize_database()
        self.syllabus_manager = SyllabusManager()

    def update_user_syllabus(self, username: str, syllabus: List[str]):
        if username in self.users:
            self.users[username]['syllabus'] = syllabus
            self._save_users()

    def delete_subject(self, username: str, subject_name: str) -> bool:
        """Delete a subject and its associated vector store collection"""
        try:
            # Get user_id from username
            user_id = self.db.get_user_id(username)
            if not user_id:
                return False
                
            # Get all subjects for the user
            subjects = self.db.get_user_subjects(user_id)
            subject = next((s for s in subjects if s['subject_name'] == subject_name), None)
            
            if subject:
                # First delete from vector database
                vector_db_deleted = self.syllabus_manager.delete_subject_collection(username, subject_name)
                
                if vector_db_deleted:
                    # Then delete from database
                    success = self.db.delete_subject(subject['id'])
                    return success
                else:
                    st.warning("Failed to delete subject data from vector database. Database not modified.")
                    return False
            return False
            
        except Exception as e:
            st.error(f"Error deleting subject: {str(e)}")
            return False

    def get_user_syllabus(self, username: str) -> List[str]:
        return self.users.get(username, {}).get('syllabus', [])

    def add_chat_history(self, username: str, message: Dict):
        if username in self.users:
            if 'chat_history' not in self.users[username]:
                self.users[username]['chat_history'] = []
            self.users[username]['chat_history'].append(message)
            self._save_users()

    def add_custom_topic(self, username: str, subject_name: str, topic: str) -> bool:
        """Add a custom topic to a subject's syllabus"""
        try:
            user_id = self.db.get_user_id(username)
            if not user_id:
                return False
                
            subjects = self.db.get_user_subjects(user_id)
            subject = next((s for s in subjects if s['subject_name'] == subject_name), None)
            
            if subject:
                # Check if topic already exists
                existing_topics = self.db.get_subject_topics(subject['id'])
                if topic not in existing_topics.get('custom_topics', []):
                    # Add the new custom topic
                    success = self.db.add_topic(subject['id'], topic, is_custom=True)
                    return success
            return False
            
        except Exception as e:
            st.error(f"Error adding custom topic: {str(e)}")
            return False
    
    def get_subject_syllabus(self, username: str, subject_name: str) -> List[str]:
        """Get syllabus topics for a specific subject using the database"""
        user_id = self.db.get_user_id(username)
        if user_id:
            subjects = self.db.get_user_subjects(user_id)
            subject = next((s for s in subjects if s['subject_name'] == subject_name), None)
            
            if subject:
                # Retrieve topics for this subject from the database
                topics = self.db.get_subject_topics(subject['id'])
                
                # Add a robust check to handle different possible return types
                if topics and isinstance(topics, list):
                    return [
                        topic['topic_name'] if isinstance(topic, dict) and 'topic_name' in topic 
                        else str(topic) 
                        for topic in topics 
                        if not (isinstance(topic, dict) and topic.get('is_custom', False))
                    ]
            
        return []

    def delete_custom_topic(self, username: str, subject_name: str, topic: str) -> bool:
        """
        Delete a custom topic from a subject
        
        Args:
            username (str): The username of the user
            subject_name (str): The name of the subject
            topic (str): The topic text to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            # Get user ID
            user_id = self.db.get_user_id(username)
            if not user_id:
                return False
                
            # Get user's subjects
            subjects = self.db.get_user_subjects(user_id)
            subject = next((s for s in subjects if s['subject_name'] == subject_name), None)
            
            if not subject:
                return False
                
            # Get subject topics
            topics = self.db.get_subject_topics(subject['id'])
            
            # Check if the topic exists in custom_topics
            custom_topics = topics.get('custom_topics', [])
            if topic in custom_topics:
                # We need to get the topic ID from the database
                connection = self.db.connection_pool.get_connection()
                cursor = connection.cursor()
                
                try:
                    # Find the topic ID where the text matches and is_custom is True
                    cursor.execute("""
                        SELECT id FROM topics 
                        WHERE subject_id = %s 
                        AND topics_name = %s 
                        AND is_custom = TRUE
                        """, (subject['id'], topic))
                    
                    result = cursor.fetchone()
                    if result:
                        topic_id = result[0]
                        success = self.db.delete_topic(topic_id)
                        return success
                        
                    return False
                    
                finally:
                    cursor.close()
                    connection.close()
                    
            return False
            
        except Exception as e:
            st.error(f"Error deleting custom topic: {str(e)}")
            return False
        
    def get_subject_topics(self, username: str, subject_name: str) -> Dict[str, List[str]]:
        """Get all topics for a specific subject, separated into syllabus and custom topics"""
        try:
            user_id = self.db.get_user_id(username)
            if not user_id:
                return {'syllabus_topics': [], 'custom_topics': []}
                
            subjects = self.db.get_user_subjects(user_id)
            subject = next((s for s in subjects if s['subject_name'] == subject_name), None)
            
            if subject:
                # Get all topics for the subject
                topics = self.db.get_subject_topics(subject['id'])
                
                # Initialize empty lists for both types of topics
                syllabus_topics = []
                custom_topics = []
                
                # Handle different possible return types from the database
                if isinstance(topics, dict):
                    # If topics is already separated into categories
                    syllabus_topics = topics.get('syllabus_topics', [])
                    custom_topics = topics.get('custom_topics', [])
                elif isinstance(topics, list):
                    # If topics is a flat list of topic dictionaries
                    for topic in topics:
                        if isinstance(topic, dict):
                            topic_name = topic.get('topic_name', '')
                            if topic_name:
                                if topic.get('is_custom', False):
                                    custom_topics.append(topic_name)
                                else:
                                    syllabus_topics.append(topic_name)
                
                return {
                    'syllabus_topics': syllabus_topics,
                    'custom_topics': custom_topics
                }
                
            return {'syllabus_topics': [], 'custom_topics': []}
            
        except Exception as e:
            st.error(f"Error getting subject topics: {str(e)}")
            return {'syllabus_topics': [], 'custom_topics': []}
    
    def get_subject_syllabus(self, username: str, subject_name: str) -> List[str]:
        """Get only syllabus topics (non-custom) for a specific subject"""
        user_id = self.db.get_user_id(username)
        if user_id:
            subjects = self.db.get_user_subjects(user_id)
            subject = next((s for s in subjects if s['subject_name'] == subject_name), None)
            
            if subject:
                # Get all topics for the subject
                topics = self.db.get_subject_topics(subject['id'])
                
                # Filter to only include non-custom topics
                return [
                    topic['topic_name'] 
                    for topic in topics 
                    if isinstance(topic, dict) and not topic.get('is_custom', False)
                ]
        
        return []
    
    def register_user(self, username: str, password: str) -> bool:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        return self.db.add_user(username, hashed_password) is not None

    def verify_user(self, username: str, password: str) -> bool:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        return self.db.verify_user(username, hashed_password)

    def add_subject(self, username: str, subject_name: str) -> bool:
        user_id = self.db.get_user_id(username)
        if user_id:
            return self.db.add_subject(user_id, subject_name) is not None
        return False

    def get_user_subjects(self, username: str) -> List[str]:
        user_id = self.db.get_user_id(username)
        if user_id:
            subjects = self.db.get_user_subjects(user_id)
            return [subject['subject_name'] for subject in subjects]
        return []

    def update_subject_syllabus(self, username: str, subject_name: str, topics: List[str]):
        user_id = self.db.get_user_id(username)
        if user_id:
            subjects = self.db.get_user_subjects(user_id)
            subject_id = next((s['id'] for s in subjects if s['subject_name'] == subject_name), None)
            if subject_id:
                # Clear existing non-custom topics
                self.db.clear_syllabus_topics(subject_id)
                
                # Add new topics
                for topic in topics:
                    self.db.add_topic(subject_id, topic, is_custom=False)
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
class AcademicChatbot:
    def __init__(self):
        
        self.llm = ChatGroq(
            temperature=0.7,
            groq_api_key = os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            )
        
        # Set token limits for context management
        self.max_context_tokens = 4000
        self.max_history_messages = 15 # Max number of messages to consider

        self.exam_mode = ExamMode(self.llm)
        self.syllabus_manager = SyllabusManager()
        self.db = DatabaseManager()
        self.academic_template = """You are an intelligent academic assistant specialized in helping students understand their course materials and syllabus content.

        Current Context:
        - Selected Subject: {subject}
        - Syllabus Topics: {syllabus}
        - Retrieved Syllabus Context: {syllabus_context}
        
        Previous conversation summary for this subject:
        {chat_summary}

        Most recent messages:
        {recent_messages}

        Guidelines for responses:
        1. If the question relates to the selected subject's syllabus, prioritize information from the syllabus context
        2. Provide detailed academic explanations with examples
        3. For general queries outside the syllabus, give accurate information while maintaining academic rigor
        4. Always cite specific syllabus topics when they're relevant to the answer
        5. Include practical examples and applications when possible

        Human: {input}
        Assistant: """

        # Template for summarization
        self.summarization_template = """Summarize the following conversation between a user and an academic assistant.
        Focus on capturing key topics discussed, important questions asked, and essential information provided.
        Keep the summary concise but informative, emphasizing main subject matter concepts.

        Conversation:
        {conversation}
        
        Summary:"""

    def _summarize_chat_history(self, history):
        """Summarize a long chat history to save context space"""
        if not history or len(history) < 5:  # Don't summarize short conversations
            return ""
            
        # Format conversation for summarization
        conversation = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in history
        ])
        
        try:
            # Use the LLM to summarize the conversation
            summarization_prompt = self.summarization_template.format(conversation=conversation)
            summary = self.llm.predict(summarization_prompt)
            return summary.strip()
        except Exception as e:
            # Fallback if summarization fails
            return f"Prior conversation covered topics related to the subject. Error summarizing: {str(e)}"
        
    def get_response(self, query: str, username: str, selected_subject: str, subject_chat_histories: Dict, use_syllabus_context: bool = True) -> str:
        try:
            user_id = self.db.get_user_id(username)
            if not user_id:
                return "Error: User not found"

            # Get subject ID if specific subject selected
            subject_id = None
            if selected_subject and selected_subject != "All":
                subjects = self.db.get_user_subjects(user_id)
                subject_id = next((s['id'] for s in subjects if s['subject_name'] == selected_subject), None)

            # Get recent chat history (now we'll split it into segments)
            chat_history = self.db.get_chat_history(user_id, subject_id, limit=30)  # Get more history

            # Split history into recent messages and older history for summarization
            recent_messages = chat_history[:3]  # Keep 3 most recent messages intact
            older_history = chat_history[5:]    # Summarize older messages

            # Format recent messages verbatim
            formatted_recent  = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}" 
                for msg in reversed(recent_messages)
            ])

            # Summarize older history if it exists
            chat_summary = ""
            if older_history:
                chat_summary = self._summarize_chat_history(older_history)

            # Get syllabus context
            syllabus_context = ""
            syllabus_topics = []
            if use_syllabus_context and selected_subject != "All":
                syllabus_context = self.syllabus_manager.get_relevant_context(
                    query, username, selected_subject
                )

                # Get syllabus topics for better context
                topics_data = st.session_state.user_manager.get_subject_topics(username, selected_subject)
                if topics_data:
                    syllabus_topics = topics_data.get('syllabus_topics', [])

            # Prepare prompt template with summarized context
            full_context = self.academic_template.format(
                subject=selected_subject,
                syllabus=", ".join(syllabus_topics[:10]),  # Limit number of topics
                syllabus_context=syllabus_context,
                chat_summary=chat_summary,
                recent_messages=formatted_recent,
                input=query
            )

            # Generate response
            response = self.llm.predict(full_context)

            # Save the interaction to chat history
            self.db.add_chat_message(user_id, subject_id, "user", query)
            self.db.add_chat_message(user_id, subject_id, "assistant", response)

            return response.strip()

        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try again."
        
def initialize_session_state():
    """Initialize all session state variables"""
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'user_manager' not in st.session_state:
        st.session_state.user_manager = UserManager()
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = AcademicChatbot()
    if 'current_subject' not in st.session_state:
        st.session_state.current_subject = None
    if 'subject_chat_histories' not in st.session_state:
        st.session_state.subject_chat_histories = {}
    if 'use_syllabus_context' not in st.session_state:
        st.session_state.use_syllabus_context = True
    if 'exam_mode' not in st.session_state:
        st.session_state.exam_mode = False
    if 'current_quiz' not in st.session_state:
        st.session_state.current_quiz = None

def main():
    st.set_page_config(page_title="AceBot", layout="wide")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for authentication
    with st.sidebar:
        st.title("AceBot")
        if st.session_state.user is None:
            tab1, tab2 = st.tabs(["Login", "Register"])
            
            with tab1:
                login_username = st.text_input("Username", key="login_username")
                login_password = st.text_input("Password", type="password", key="login_password")
                if st.button("Login"):
                    if st.session_state.user_manager.verify_user(login_username, login_password):
                        st.session_state.user = login_username
                        st.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")

            with tab2:
                reg_username = st.text_input("Username", key="reg_username")
                reg_password = st.text_input("Password", type="password", key="reg_password")
                if st.button("Register"):
                    if st.session_state.user_manager.register_user(reg_username, reg_password):
                        st.success("Registration successful!")
                    else:
                        st.error("Username already exists")
        else:
            st.write(f"Welcome, {st.session_state.user}!")
            if st.button("Logout"):
                # Clear all session state
                st.session_state.user = None
                st.session_state.subject_chat_histories = {}
                st.session_state.current_subject = None
                st.session_state.messages = []
                st.rerun()

    # Main content
    if st.session_state.user is None:
        # Display welcome image and message when not logged in
        col1, col2 = st.columns([2, 2])
        
        with col1:
            st.title("Welcome to AceBot")
            st.subheader("Your AI Academic Assistant")
            st.write("""
            AceBot helps you study smarter with:
            - Subject-specific chat assistance
            - Syllabus management
            - Exam preparation tools
            - Personalized learning
            
            Login or register to get started!
            """)
        
        with col2:
            # Display welcome image
            st.image("login.jpg", use_column_width=True)
    
    else:
        tab1, tab2, tab3 = st.tabs(["Chat", "Subjects & Syllabus", "Exam Mode"])

        with tab1:
            st.header("Academic Chat")
            
            # Add subject selector to chat interface
            user_subjects = st.session_state.user_manager.get_user_subjects(st.session_state.user)
            
            if user_subjects:
                new_subject = st.selectbox(
                    "Select Subject",
                    ["All"] + user_subjects,
                    key="chat_subject"
                )

                # Handle subject change
                if st.session_state.current_subject != new_subject:
                    if 'messages' in st.session_state:
                        st.session_state.messages = []
                    st.session_state.current_subject = new_subject

            # Add voice chat UI
            add_voice_chat_ui(st)

            # Initialize subject-specific chat history if needed
            if st.session_state.current_subject not in st.session_state.subject_chat_histories:
                st.session_state.subject_chat_histories[st.session_state.current_subject] = []

            # Syllabus context switch
            st.session_state.use_syllabus_context = st.toggle(
                "Use Syllabus Context",
                value=st.session_state.use_syllabus_context,
                help="Toggle to enable/disable syllabus-based responses"
            )
            
            # Display current subject's chat history
        current_history = st.session_state.subject_chat_histories.get(st.session_state.current_subject, [])
        for message in current_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input("Ask anything..."):
            with st.chat_message("user"):
                st.write(prompt)

            # Add user message to current subject's history
            current_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                response = st.session_state.chatbot.get_response(
                    prompt,
                    st.session_state.user,
                    st.session_state.current_subject,
                    st.session_state.subject_chat_histories,
                    st.session_state.use_syllabus_context
                )
                st.write(response)

            # Add assistant response to current subject's history
            current_history.append({"role": "assistant", "content": response})
            
            # Update the subject's chat history in session state
            st.session_state.subject_chat_histories[st.session_state.current_subject] = current_history

        with tab2:
            st.header("Subjects & Syllabus Management")
            
            # Subject management section
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Add New Subject")
                new_subject = st.text_input("Subject Name")
                if st.button("Add Subject"):
                    if new_subject:
                        if st.session_state.user_manager.add_subject(st.session_state.user, new_subject):
                            st.success(f"Added subject: {new_subject}")
                            st.rerun()
                        else:
                            st.error("Subject already exists")

            with col2:
                st.subheader("Manage Existing Subjects")
                user_subjects = st.session_state.user_manager.get_user_subjects(st.session_state.user)
                if user_subjects:
                    selected_subject = st.selectbox("Select Subject to Manage", user_subjects)
                    
                    if st.button("Delete Subject", key="delete_subject"):
                        if st.session_state.user_manager.delete_subject(st.session_state.user, selected_subject):
                            st.success(f"Deleted subject: {selected_subject} and its associated data")
                            st.rerun()
                        else:
                            st.error("Failed to delete subject completely. Please try again.")
            
            # Show syllabus management for selected subject
            if user_subjects and selected_subject:
                st.subheader(f"Manage Syllabus for {selected_subject}")
                syllabus_manager = SyllabusManager()
                syllabus_manager.display_subject_syllabus(st.session_state.user, selected_subject, st)
                
                # File upload section
                uploaded_file = st.file_uploader(
                    "Upload Syllabus (PDF or Image)", 
                    type=["pdf", "png", "jpg", "jpeg"]
                )
                
                if uploaded_file is not None:
                    if st.button("Process File"):
                        with st.spinner("Processing file..."):
                            topics = st.session_state.chatbot.syllabus_manager.process_file(
                                uploaded_file,
                                st.session_state.user,
                                selected_subject
                            )
                            st.success("File processed successfully!")
                            if topics:
                                st.session_state.user_manager.update_subject_syllabus(
                                    st.session_state.user,
                                    selected_subject,
                                    topics
                                )
                                st.success("File processed successfully!")
                                st.rerun()

                # Manual syllabus input
                current_syllabus = st.session_state.user_manager.get_subject_syllabus(
                    st.session_state.user,
                    selected_subject
                )
                syllabus_text = "\n".join(current_syllabus)
                
                new_syllabus = st.text_area(
                    "Edit syllabus (one topic per line):",
                    value=syllabus_text,
                    height=300
                )
                
                if st.button("Save Syllabus"):
                    topics = [topic.strip() for topic in new_syllabus.split('\n') if topic.strip()]
                    st.session_state.user_manager.update_subject_syllabus(
                        st.session_state.user,
                        selected_subject,
                        topics
                    )
                    st.success("Syllabus updated successfully!")
                    st.rerun()
                        
        with tab3:
            st.header("Exam Mode")
            
            user_subjects = st.session_state.user_manager.get_user_subjects(st.session_state.user)
            if not user_subjects:
                st.warning("Please add subjects and their syllabi first!")
            else:
                selected_subject = st.selectbox("Select Subject", user_subjects, key="exam_subject")
                
                # Add previous papers upload section
                st.subheader("Upload Previous Year Papers")
                uploaded_papers = st.file_uploader(
                    "Upload previous year question papers (PDF)",
                    type=['pdf'],
                    accept_multiple_files=True
                )
                
                if uploaded_papers:
                    if st.button("Process Papers"):
                        with st.spinner("Processing papers..."):
                            results = st.session_state.chatbot.exam_mode.process_previous_papers(
                                st.session_state.user,
                                selected_subject,
                                uploaded_papers
                            )
                            
                            if results["success"]:
                                st.success(f"Successfully processed {len(results['processed_files'])} files!")
                                st.write("Processed files:", ", ".join(results['processed_files']))
                                st.write(f"Total chunks created: {results['total_documents']}")
                                if results.get('analysis'):
                                    st.write("Analysis:", results['analysis'])
                                    
                                if results['failed_files']:
                                    st.warning("Some files had issues:")
                                    for fail in results['failed_files']:
                                        st.write(f"- {fail['filename']}: {fail['error']}")
                            else:
                                st.error(f"Processing failed: {results['error']}")
                                if results['failed_files']:
                                    st.write("Failed files:")
                                    for fail in results['failed_files']:
                                        st.write(f"- {fail['filename']}: {fail['error']}")
                
                # Get both syllabus and custom topics
                topics_data = st.session_state.user_manager.get_subject_topics(
                    st.session_state.user,
                    selected_subject
                )
                
                # Debug output to see what we're getting
                st.status(f"Topics data received: {topics_data}")
                
                # Custom topic input
                with st.expander("Add Custom Topic"):
                    new_topic = st.text_input("Enter new topic:")
                    if st.button("Add Topic"):
                        if new_topic:
                            if st.session_state.user_manager.add_custom_topic(
                                st.session_state.user,
                                selected_subject,
                                new_topic
                            ):
                                st.success(f"Added custom topic: {new_topic}")
                                st.rerun()
                            else:
                                st.error("Topic already exists or failed to add")
                
                # Topic selection
                topic_source = st.radio(
                    "Select topic source:",
                    ["Syllabus Topics", "Custom Topics"],
                    key="topic_source"
                )
                
                # Get the appropriate list of topics based on selection
                available_topics = (topics_data['syllabus_topics'] 
                                if topic_source == "Syllabus Topics" 
                                else topics_data['custom_topics'])
                
                if not available_topics:
                    st.warning(f"No {topic_source.lower()} available for {selected_subject}. Please add some topics first!")
                else:
                    selected_topic = st.selectbox(
                        "Select Topic for Quiz", 
                        available_topics,
                        key="topic_selector"
                    )
                    
                    # Delete custom topic option
                    if topic_source == "Custom Topics" and selected_topic:
                        if st.button("Delete Selected Topic"):
                            if st.session_state.user_manager.delete_custom_topic(
                                st.session_state.user,
                                selected_subject,
                                selected_topic
                            ):
                                st.success(f"Deleted topic: {selected_topic}")
                                st.rerun()
                            else:
                                st.error("Failed to delete topic")
                    
                    # Generate quiz only if a topic is selected
                    if selected_topic and st.button("Generate New Question"):
                        with st.spinner("Generating question..."):
                            st.session_state.current_quiz = st.session_state.chatbot.exam_mode.generate_adaptive_quiz(
                                selected_topic,
                                st.session_state.user,
                                selected_subject
                            )
                            if st.session_state.current_quiz is None:
                                st.error("Failed to generate quiz. Please try again.")
                    
                    # Display quiz with performance tracking
                    if hasattr(st.session_state, 'current_quiz') and st.session_state.current_quiz:
                        quiz = st.session_state.current_quiz
                        st.write(f"**Question:** {quiz['question']}")
                        st.write(f"**Level:** {quiz.get('level', 'Medium')}")
                        st.write(f"**Category:** {quiz.get('category', 'General')}")
                        
                        options_display = [opt.split(')', 1)[1].strip() for opt in quiz['options']]
                        selected_answer = st.radio(
                            "Select your answer:",
                            options_display,
                            key="quiz_answer"
                        )
                        
                        if st.button("Check Answer"):
                            selected_index = options_display.index(selected_answer)
                            correct_letter = quiz['correct']
                            correct_index = ord(correct_letter) - ord('A')
                            is_correct = selected_index == correct_index
                            
                            if is_correct:
                                st.success("Correct! 🎉")
                            else:
                                st.error(f"Incorrect. The correct answer was {correct_letter}.")
                            
                            st.write("**Explanation:**")
                            st.write(quiz['explanation'])
                            
                            # Track performance
                            st.session_state.chatbot.exam_mode.track_performance(
                                st.session_state.user,
                                selected_subject,
                                selected_topic,
                                quiz,
                                chr(ord('A') + selected_index),
                                is_correct
                            )
                    
                    # Add weak areas analysis section
                    st.subheader("Performance Analysis")
                    if st.button("Show Weak Areas"):
                        weak_areas = st.session_state.chatbot.exam_mode.get_weak_areas(
                            st.session_state.user,
                            selected_subject
                        )
                        
                        if weak_areas:
                            st.write("Here are your areas that need improvement:")
                            for area in weak_areas:
                                with st.expander(f"Topic: {area['topic']} - Accuracy: {area['accuracy']:.1f}%"):
                                    # Overall Statistics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Overall Accuracy", f"{area['accuracy']:.1f}%")
                                    with col2:
                                        st.metric("Recent Trend", f"{area['recent_trend']:.1f}%")
                                    with col3:
                                        st.metric("Total Questions", area['total_questions'])
                                    
                                    # Weak Subtopics
                                    if area['weak_subtopics']:
                                        st.subheader("Weak Subtopics")
                                        for subtopic in area['weak_subtopics']:
                                            st.write(f"- {subtopic['name']}: {subtopic['accuracy']:.1f}% ({subtopic['attempts']} attempts)")
                                    
                                    # Recent Questions
                                    st.subheader("Recent Questions")
                                    for q in area['recent_questions']:
                                        status = "✅" if q['correct'] else "❌"
                                        st.write(f"{status} {q['question']}")
                                        st.write(f"   Level: {q['level']} | Category: {q['category']}")
                                    
                                    # Difficulty Distribution
                                    st.subheader("Performance by Difficulty")
                                    diff_data = area['difficulty_distribution']
                                    st.bar_chart(diff_data)
                                    
                                    # Categories
                                    st.subheader("Categories Covered")
                                    st.write(", ".join(area['categories']))
                        else:
                            st.info("No weak areas identified yet. Keep practicing!")

                    st.subheader("Probable Questions for Exam")
                    # Add question pattern analysis
                    if st.button("Analyze Question Patterns"):
                        with st.spinner("Analyzing previous year question patterns..."):
                            patterns = st.session_state.chatbot.exam_mode.analyze_question_patterns(
                                st.session_state.user,
                                selected_subject
                            )
                            if patterns:
                                st.write("### Question Pattern Analysis")
                                st.write(patterns["raw_analysis"])

                    # Generate probable questions
                    col1, col2 = st.columns(2)
                    with col1:
                        num_questions = st.number_input("Number of questions to generate", 5, 20, 10)
                    with col2:
                        question_type = st.selectbox("Question Type", ["All", "Theory", "Numerical"])

                    if st.button("Generate Probable Questions"):
                        with st.spinner("Analyzing previous papers and generating probable questions..."):
                            probable_questions = st.session_state.chatbot.exam_mode.generate_probable_questions(
                                st.session_state.user,
                                selected_subject,
                                num_questions=num_questions
                            )
                            
                            if probable_questions:
                                # Filter questions based on selected type
                                if question_type != "All":
                                    probable_questions = [q for q in probable_questions if q["type"] == question_type]
                                
                                for i, q in enumerate(probable_questions, 1):
                                    with st.expander(f"Question {i}: {q['type']} - {q['marks']} Marks ({q['difficulty']})"):
                                        # Create columns for main content and notes
                                        col1, col2 = st.columns([2, 1])
                                        
                                        with col1:
                                            st.markdown(f"**Question:**\n{q['question_text']}")
                                            st.markdown("---")
                                            st.markdown(f"**Solution:**\n{q['solution']}")
                                            st.markdown("---")
                                            st.markdown(f"**Key Points:**\n{q['key_points']}")
                                            st.markdown(f"**Previous Year Pattern:**\n{q['previous_pattern']}")
                                            st.markdown(f"**Topics Covered:**\n{q['topics_covered']}")
                                        
                                        with col2:
                                            st.markdown("**📝 Examiner's Notes**")
                                            st.markdown(f"{q['examiner_notes']}")
                            else:
                                st.error("Failed to generate probable questions. Please ensure you have uploaded previous year papers.")

                    # Optional: Add topic-wise question generation
                    if st.toggle("Generate Topic-wise Questions"):
                        topics = st.session_state.user_manager.get_subject_topics(
                            st.session_state.user,
                            selected_subject
                        )
                        questions_per_topic = st.slider("Questions per topic", 1, 5, 3)
                        topic_type = st.selectbox("Topic Question Type", ["All", "Theory", "Numerical"], key="topic_type")
                        
                        if st.button("Generate Topic-wise Questions"):
                            for topic in topics:
                                st.subheader(f"Topic: {topic}")
                                with st.spinner(f"Generating questions for {topic}..."):
                                    questions = st.session_state.chatbot.exam_mode.generate_probable_questions(
                                        st.session_state.user,
                                        selected_subject,
                                        num_questions=questions_per_topic
                                    )
                                    
                                    # Filter questions based on selected type
                                    if topic_type != "All":
                                        questions = [q for q in questions if q["type"] == topic_type]
                                    
                                    for i, q in enumerate(questions, 1):
                                        with st.expander(f"Question {i}: {q['type']} - {q['marks']} Marks ({q['difficulty']})"):
                                            # Create columns for main content and notes
                                            col1, col2 = st.columns([2, 1])
                                            
                                            with col1:
                                                st.markdown(f"**Question:**\n{q['question_text']}")
                                                st.markdown("---")
                                                st.markdown(f"**Solution:**\n{q['solution']}")
                                                st.markdown("---")
                                                st.markdown(f"**Key Points:**\n{q['key_points']}")
                                                st.markdown(f"**Previous Year Pattern:**\n{q['previous_pattern']}")
                                                st.markdown(f"**Topics Covered:**\n{q['topics_covered']}")
                                            
                                            with col2:
                                                st.markdown("**📝 Examiner's Notes**")
                                                st.markdown(f"{q['examiner_notes']}")

if __name__ == "__main__":
    main()