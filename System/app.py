# Copyright (c) 2025 RIAOR SYSTEM LTD
# All rights reserved.
import os
import logging
import random
import asyncio
import sys
from datetime import datetime
from typing import Optional, Tuple, Dict, List
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh
import torch
import warnings
from sqlalchemy import text
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import database and models
from database import SessionLocal, init_db, encrypt_data
from models import (
    ComplaintStatus, add_complaint, add_customer,
    get_customer_by_account, get_all_complaints,
    update_complaint_status, get_live_complaint_stats,
    Complaint, Customer, Routing
)

# Initialize Afriastalking SMS
try:
    import africastalking
except ImportError:
    africastalking = None
    logging.warning("Africastalking package not installed")

# Configure environment and logging
load_dotenv()
logger = logging.getLogger(__name__)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Failed to init")

# Configure Streamlit
st.set_page_config(
    page_title="PENNYYPAL SYSTEM",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("PENNYPAL SYSTEM")
st.sidebar.header("📝 File a complaint")

# Department mapping with phone numbers
department_map = {
    "credit_card": {
        "name": "Card Services",
        "contact": "+25411898890"
    },
    "retail_banking": {
        "name": "Customer Support", 
        "contact": "+254793781276"
    },
    "credit_reporting": {
        "name": "Credit Department",
        "contact": "+254769138014"
    },
    "mortgages_and_loans": {
        "name": "Loans Department",
        "contact": "+254719177569"
    },
    "debt_collection": {
        "name": "Collections",
        "contact": "+254782996878"
    }
}

# Initialize Afriastalking SMS
def initialize_sms_service():
    if africastalking and os.getenv('AT_API_KEY'):
        africastalking.initialize(
            username=os.getenv('AT_USERNAME'),
            api_key=os.getenv('AT_API_KEY')
        )
        return africastalking.SMS
    return None

sms_service = initialize_sms_service()

def send_sms(recipient: str, message: str) -> bool:
    """Send SMS using Afriastalking"""
    if not sms_service:
        logger.warning("SMS service not available")
        return False
    
    try:
        response = sms_service.send(message, [recipient])
        logger.info(f"SMS sent to {recipient}: {response}")
        return True
    except Exception as e:
        logger.error(f"Failed to send SMS: {str(e)}")
        return False

def fix_event_loop():
    """Fix for Windows event loop policy"""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

fix_event_loop()

# Application initialization
def initialize_app():
    """Initialize application components"""
    try:
        init_db()
        
        if 'last_update_time' not in st.session_state:
            st.session_state.last_update_time = None

        # Verify database connection
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.critical(f"Application initialization failed: {str(e)}")
        st.error(f"Failed to initialize application: {str(e)}")
        raise

# Complaint classifier using DistilBERT
class ComplaintClassifier:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the classification model"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            model_name = "d9lph8n/saved_DistilBert_model"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, token=os.getenv("HUGGINGFACE_TOKEN"))
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=5, token=os.getenv("HUGGINGFACE_TOKEN"))
            self.model.eval()
            logger.info("NLP model loaded successfully")
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            st.warning("Classification feature is currently unavailable")

    def classify_text(self, text: str) -> str:
        """Classify complaint text into categories"""
        if not text or not self.model:
            return "unknown"

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()
            return list(department_map.keys())[predicted_label]
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return "unknown"

# UI Components
def configure_ui():
    """Configure Streamlit UI styles"""
    st.markdown("""
    <style>
        .stApp { background-color: #0A192F; }
        .css-18e3th9 { 
            padding: 1rem; 
            background-color: #3D0C02; 
            border-radius: 10px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .complaint-card {
            background-color: #112240;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-left: 4px solid #4e73df;
        }
        .metric-card {
            background-color: #191970;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

def show_document_animation():
    """Falling financial documents animation"""
    docs = ["🧾", "📄", "📑", "📋"]
    st.markdown("".join([
        f"""<div class="doc" style="left:{random.randint(10, 90)}%;
            animation-delay:{random.uniform(0, 1)}s;
            animation-duration:{random.uniform(1, 2)}s;
            transform: rotate({random.randint(-15, 15)}deg);
            ">{random.choice(docs)}</div>"""
        for _ in range(8)
    ]) + """
    <style>
    @keyframes doc-fall {
        0% { transform: translateY(-100px) rotate(0deg); opacity: 0; }
        100% { transform: translateY(100vh) rotate(30deg); opacity: 1; }
    }
    .doc {
        position: fixed;
        font-size: 24px;
        animation: doc-fall linear forwards;
        z-index: 1000;
        pointer-events: none;
    }
    </style>
    """, unsafe_allow_html=True)

# Complaint Submission
def complaint_submission_form(classifier: ComplaintClassifier) -> bool:
    """Render complaint submission form"""
    with st.form("complaint_form", clear_on_submit=True):
        st.subheader("📩 Submit a Complaint")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name:", key="name_input")
        with col2:
            phone = st.text_input(
                "Phone Number:", 
                placeholder="07XX XXX XXX",
                help="Format: 07XXXXXXXX or +2547XXXXXXXX"
            )
        
        account_number = st.text_input("Account Number:")
        complaint_text = st.text_area("Describe your issue:", height=200)

        # Classify complaint
        category = "unknown"
        if complaint_text and len(complaint_text) > 10:
            with st.spinner("Analyzing complaint..."):
                category = classifier.classify_text(complaint_text)
            if category != "unknown":
                st.success(f"🔍 Detected Category: {department_map[category]['name']}")

        if st.form_submit_button("Submit Complaint"):
            if validate_complaint(name, phone, account_number, complaint_text):
                success = handle_complaint_submission(
                    name=name,
                    phone=phone,
                    account=account_number,
                    text=complaint_text,
                    category=category
                )
                if success:
                    st.balloons()
                    show_document_animation()
                    st.success("""
                    **Complaint Submitted Successfully!**  
                    We've received your complaint and will process it shortly.  
                    You'll receive an SMS confirmation.
                    """)
                    
                    # Auto-refresh the page after 3 seconds
                    st.session_state.should_rerun = True
                    time.sleep(3)
                    st.rerun()
                return success


    return False

def validate_complaint(name: str, phone: str, account: str, text: str) -> bool:
    """Validate complaint form inputs"""
    valid = True
    
    if len(name.strip()) < 2:
        st.error("Please enter a valid name (minimum 2 characters)")
        valid = False
        
    phone = phone.strip().replace(" ", "")
    if not (phone.startswith(('+254', '07')) and len(phone) in (10, 12)):
        st.error("Please enter a valid Kenyan phone number")
        valid = False
        
    if not account.strip():
        st.error("Please enter your account number")
        valid = False
        
    if len(text) < 30:
        st.error("Please provide more details about your issue")
        valid = False
        
    return valid

def format_phone(phone: str) -> str:
    """Format phone number to E.164 standard"""
    phone = phone.strip().replace(" ", "")
    if phone.startswith('0') and len(phone) == 10:
        return f"+254{phone[1:]}"
    elif not phone.startswith('+254') and len(phone) == 12:
        return f"+254{phone}"
    return phone

def handle_complaint_submission(name: str, phone: str, account: str, 
                              text: str, category: str) -> bool:
    """Process complaint submission"""
    try:
        formatted_phone = format_phone(phone)
        if not formatted_phone.startswith('+254'):
            st.error("Invalid phone number format")
            return False

        encrypted_account = encrypt_data(account)

        with SessionLocal() as session:
            # Find or create customer
            customer = session.query(Customer).filter(
                Customer.account_number == encrypt_data(account)).first()
            
            if not customer:
                customer = Customer(
                    name=name,
                    phone_number=encrypt_data(formatted_phone),
                    phone_display=f"**_**{formatted_phone[-4:]}",
                    account_number=encrypted_account,
                    account_display=f"****{account[-4:]}"
                )
                session.add(customer)
                session.flush()

            # Create complaint
            complaint = Complaint(
                customer_id=customer.id,
                complaint_text=text,
                category=category,
                status=ComplaintStatus.SUBMITTED.value
            )
            session.add(complaint)
            session.flush()

            # Create routing
            routing = Routing(
                customer_id=customer.id,
                complaint_id=complaint.id,
                customer_account_number=customer.account_number,
                account_display=f"****{account[-4:]}",
                route_name=department_map[category]["name"],
                created_at=datetime.utcnow()
            )
            session.add(routing)
            session.commit()

            # Send confirmation SMS
            send_confirmation_sms("Dear customer,thank you for your trust,the issue will be handled in 24 hours time.")
            send_department_alert(category, complaint.id)

            # UI feedback
            st.session_state.last_update_time = datetime.now()
            st.toast("Complaint received!", icon="✅")
            st.success(f"""
            **Complaint Submitted Successfully!**
            - Case Number: #{complaint.id}
            - Category: {department_map[category]['name']}
            """)
            show_document_animation()
            return True

    except Exception as e:
        logger.error(f"Complaint submission failed: {str(e)}", exc_info=True)
        st.error("Failed to process complaint")
        return False

def send_confirmation_sms(phone: str, case_id: int, category: str) -> None:
    """Send confirmation SMS to complainant"""
    message = (
        f"Thank you for your complaint (Case #{case_id}). "
        f"Your issue has been routed to {department_map[category]['name']}. "
        "We'll contact you shortly."
    )
    if send_sms(phone, message):
        logger.info(f"Confirmation SMS sent to {phone}")
    else:
        logger.warning(f"Failed to send confirmation SMS to {phone}")

def send_department_alert(category: str, case_id: int) -> None:
    """Send alert to department about new complaint"""
    dept = department_map.get(category)
    if dept and dept["contact"]:
        message = (
            f"New complaint received (Case #{case_id}). "
            f"Category: {dept['name']}. Please check the system."
        )
        if send_sms(dept["contact"], message):
            logger.info(f"Alert sent to {dept['name']} at {dept['contact']}")
        else:
            logger.warning(f"Failed to send alert to {dept['name']}")

# Complaint Display
def display_complaint_history():
    """Display complaint history with filters"""
    st.subheader("📜 Complaint History")
    
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All"] + [status.value for status in ComplaintStatus]
        )
    with col2:
        search_term = st.text_input("Search complaints")
    
    try:
        complaints = get_all_complaints(
            search_filter=search_term if search_term else None,
            status_filter=status_filter if status_filter != "All" else None
        )
        
        if not complaints:
            st.info("No complaints found")
            return
            
        for complaint in complaints:
            with st.expander(f"Complaint #{complaint['id']} - {complaint['status']}"):
                cols = st.columns([1, 3])
                with cols[0]:
                    st.markdown(f"""
                    **Category**: {complaint['category'].replace('_', ' ').title()}  
                    **Status**: {complaint['status']}  
                    **Date**: {complaint['created_at'].split('T')[0] if 'T' in complaint['created_at'] else complaint['created_at']}
                    """)
                    
                    new_status = st.selectbox(
                        "Update Status",
                        [s.value for s in ComplaintStatus],
                        index=[s.value for s in ComplaintStatus].index(complaint['status']),
                        key=f"status_update_{complaint['id']}"
                    )
                    
                    if new_status != complaint['status']:
                        if st.button("Update", key=f"update_{complaint['id']}"):
                            if update_complaint_status(complaint['id'], new_status):
                                st.success("Status updated!")
                                st.rerun()
                            else:
                                st.error("Update failed")
                
                with cols[1]:
                    st.markdown(f"**Description**:\n\n{complaint['complaint_text']}")
                    if complaint.get('customer'):
                        st.markdown(f"""
                        **Account**: {complaint['customer']['account_display']}  
                        **Phone**: {complaint['customer']['phone_display']}
                        """)
                    
    except Exception as e:
        st.error(f"Failed to load complaints: {str(e)}")
        logger.error(f"Complaint loading error: {str(e)}")

def display_complaint_stats():
    """Display complaint statistics dashboard"""
    st.subheader("Live Complaint Dashboard")
    
    try:
        stats = get_live_complaint_stats()
        if not stats:
            st.warning("No complaint data available")
            return

        cols = st.columns(5)
        status_colors = {
            "Submitted": "#4e73df",
            "In Progress": "#f6c23e",
            "Resolved": "#1cc88a",
            "Pending": "#858796",
            "Deleted": "#e74a3b"
        }
        
        for i, (status, count) in enumerate(stats.items()):
            with cols[i]:
                st.markdown(
                    f"""<div style="background-color:#112240; padding:15px; border-radius:10px; 
                        border-left:5px solid {status_colors.get(status, '#000')}">
                        <h3 style="color:{status_colors.get(status, '#000')}; margin:0; font-size:14px;">
                            {status.upper()}
                        </h3>
                        <h2 style="color:white; margin:0; font-size:24px;">
                            {count}
                        </h2>
                    </div>""",
                    unsafe_allow_html=True
                )
        
        # Pie chart
        chart_data = pd.DataFrame({
            "Status": list(stats.keys()),
            "Count": list(stats.values())
        })
        
        fig = px.pie(
            chart_data,
            names="Status",
            values="Count",
            hole=0.3,
            color="Status",
            color_discrete_map=status_colors
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=False, margin=dict(t=25, b=0, l=0, r=0), height=200)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Failed to load statistics: {str(e)}")

# File watcher
class DBFileHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.src_path.endswith('.db-journal'):
            st.session_state.last_update_time = datetime.now().strftime("%H:%M:%S")

def configure_file_watcher() -> Optional[Observer]:
    """Configure file watcher for database changes"""
    try:
        current_dir = os.path.normpath(os.path.dirname(__file__))
        if not os.path.exists(current_dir):
            logger.warning(f"Watchdog path doesn't exist: {current_dir}")
            return None
            
        event_handler = DBFileHandler()
        observer = Observer()
        observer.schedule(event_handler, path=current_dir, recursive=False)
        observer.start()
        return observer
    except Exception as e:
        logger.warning(f"Could not start file watcher: {str(e)}")
        return None

# Main application
def main():
    """Main application entry point"""
    observer = None
    try:
        initialize_app()
        configure_ui()

        # Configure file watcher if not in Streamlit sharing
        if not st.runtime.exists():
            observer = configure_file_watcher()

        # Auto-refresh
        refresh_rate = st.sidebar.slider(
            "Auto-refresh (minutes)",
            min_value=1, max_value=60, value=5
        )
        st_autorefresh(interval=refresh_rate * 60 * 1000)
        
        # Sidebar info
        # Sidebar info
        with st.sidebar:
            try:
                with SessionLocal() as session:
                    last_update = session.query(Complaint.updated_at)\
                    .order_by(Complaint.updated_at.desc())\
                    .first()
                    if last_update and last_update[0]:
                        last_update_str = last_update[0].strftime("%b %d, %Y %I:%M %p")
                        st.markdown(f"""<p style='font-size:16px;'><b>Last Update:</b>
                         {last_update_str}</p>
                         """, unsafe_allow_html=True)

                    else:
                        st.metric("Last Complaint Update", last_update_str)
            except Exception as e:
                logger.error(f"Database connection error: {str(e)}")
            
        # Main content
        st.title("📢 Complaint Management System")
        st.caption("Efficiently manage and track customer complaints")
        
        classifier = ComplaintClassifier()
        
        col1, col2 = st.columns([2, 1], gap="large")
        with col1:
            if complaint_submission_form(classifier):
                st.balloons()  
                st.session_state.last_update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.rerun()

            display_complaint_history()
            
        with col2:
            display_complaint_stats()
            
    except Exception as e:
        logger.critical(f"Application error: {str(e)}", exc_info=True)
        st.error("A critical error occurred. Please contact support.")
        st.error(f"Technical details: {str(e)}")
    finally:
        if observer:
            try:
                observer.stop()
                observer.join(timeout=5)
            except Exception as e:
                logger.warning(f"Error stopping observer: {str(e)}")

if __name__ == "__main__":
    if sys.platform == "win32":
        torch.multiprocessing.freeze_support()
   
    fix_event_loop()
    try:
        main()
    except Exception as e:
        logger.critical(f"Application crashed: {str(e)}", exc_info=True)
        st.error("The application encountered a critical error. Please try again later.")
        st.stop()