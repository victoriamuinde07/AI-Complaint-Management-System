import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from enum import Enum
from sqlalchemy import (
    Column, Integer, String, Text, DateTime,
    ForeignKey, Index, func
)
from sqlalchemy.orm import relationship, Session
from database import db_session, get_encryption_service,init_db
from database import Base, SessionLocal, encrypt_data, decrypt_data
from cryptography.fernet import Fernet
from sqlalchemy import or_

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize encryption
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY')
if not ENCRYPTION_KEY:
    raise ValueError("ENCRYPTION_KEY environment variable must be set")

encryption = get_encryption_service()


class ComplaintStatus(str, Enum):
    SUBMITTED = "Submitted"
    IN_PROGRESS = "In Progress"
    RESOLVED = "Resolved"
    DELETED = "Deleted"
    PENDING = "Pending"


@contextmanager
def session_scope():
    
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()


class Customer(Base):
    __tablename__ = "customers"

    __table_args__ = (
        Index('idx_customer_name', 'name'),
        {'extend_existing': True}
    )

    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    phone_number = Column(String(512))
    phone_display = Column(String(15))
    account_number = Column(String(512))
    account_display = Column(String(10))

    complaints = relationship("Complaint", back_populates="customer")
    routings = relationship("Routing", back_populates="customer")

    def to_dict(self, include_sensitive=False) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "name": self.name,
            "phone_display": self.phone_display or f"**_**{decrypt_data(self.phone_number)[-4:]}",
            "account_display": self.account_display or f"****{decrypt_data(self.account_number)[-4:]}"
        }
        if include_sensitive:
            try:
                data.update({
                    "phone_number": decrypt_data(self.phone_number),
                    "account_number": decrypt_data(self.account_number)
                })
            except Exception as e:
                logging.error(f"Decryption failed: {str(e)}")
                data.update({
                    "phone_number": "[DECRYPTION_ERROR]",
                    "account_number": "[DECRYPTION_ERROR]"
                })
        return data


class Complaint(Base):
    __tablename__ = "complaints"
    __table_args__ = (
        Index('idx_complaint_status_date', 'status', 'created_at'),
    )

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), index=True)
    complaint_text = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)
    status = Column(String(50), default=ComplaintStatus.SUBMITTED.value)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow,
                        onupdate=datetime.utcnow)

    customer = relationship("Customer", back_populates="complaints")
    routings = relationship("Routing", back_populates="complaint")

    def to_dict(self, include_relationships=False) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "customer_id": self.customer_id,
            "complaint_text": self.complaint_text,
            "category": self.category,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        if include_relationships:
            data["customer"] = self.customer.to_dict() if self.customer else None
            data["routings"] = [r.to_dict() for r in self.routings]
        return data


class Routing(Base):
    __tablename__ = "routings"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False)
    complaint_id = Column(Integer, ForeignKey("complaints.id"), index=True)
    customer_account_number = Column(String(512), nullable=False)
    account_display = Column(String(10))
    route_name = Column(String(255), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime, index=True)

    customer = relationship("Customer", back_populates="routings")
    complaint = relationship("Complaint", back_populates="routings")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "complaint_id": self.complaint_id,
            "route_name": self.route_name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "account_display": self.account_display
        }

# CRUD Operations


def add_customer(name: str, phone_number: str, account_number: str) -> Optional[int]:
    """Add a new customer with encrypted sensitive data"""
    if not all([name, phone_number, account_number]):
        logging.error("Missing required customer fields")
        return None

    try:
        with session_scope() as session:
            encrypted_account = encrypt_data(account_number)
            encrypted_phone = encrypt_data(phone_number)

            existing = session.query(Customer).filter_by(
                account_number=encrypted_account
            ).first()

            if existing:
                logging.info(
                    f"Customer with account {account_number[-4:]} already exists")
                return existing.id

            new_customer = Customer(
                name=name,
                phone_number=encrypted_phone,
                phone_display=f"**_**{phone_number[-4:]}",
                account_number=encrypted_account,
                account_display=f"****{account_number[-4:]}"
            )
            session.add(new_customer)
            session.flush()
            return new_customer.id
    except Exception as e:
        logging.error(f"Failed to add customer: {str(e)}")
        return None


def add_complaint(
    customer_id: int,
    complaint_text: str,
    category: str,
    account_number: str,
    status: str = ComplaintStatus.SUBMITTED.value
) -> Optional[int]:
    try:
        with session_scope() as session:
            customer = session.query(Customer).get(customer_id)
            if not customer:
                logging.error(f"Customer {customer_id} not found")
                return None

           
            new_complaint = Complaint(
                customer_id=customer_id,
                complaint_text=complaint_text,
                category=category,
                status=status
            )
            session.add(new_complaint)
            session.flush()  

            department_map = {
                "credit_card": "Card Services",
                "retail_banking": "Customer Support",
                "credit_reporting": "Credit Department",
                "mortgages_and_loans": "Loans Department",
                "debt_collection": "Collections"
            }

           
            routing = Routing(
                customer_id=customer_id,
                complaint_id=new_complaint.id,  
                customer_account_number=customer.account_number,
                account_display=customer.account_display,
                route_name=department_map.get(category.lower(), "General Support")
            )
            session.add(routing)
            return new_complaint.id

    except Exception as e:
        logging.error(f"Failed to add complaint: {str(e)}", exc_info=True)
        return None

def get_complaints_by_customer(customer_id: int) -> List[Dict[str, Any]]:
    """Get all complaints for a specific customer"""
    try:
        with session_scope() as session:
            complaints = session.query(Complaint)\
                .filter_by(customer_id=customer_id)\
                .order_by(Complaint.created_at.desc())\
                .all()
            return [c.to_dict(include_relationships=True) for c in complaints]
    except Exception as e:
        logging.error(
            f"Failed to get complaints for customer {customer_id}: {str(e)}")
        return []


def get_all_complaints(
    search_filter: Optional[str] = None,
    status_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get all complaints with optional filters"""
    try:
        with session_scope() as session:
            query = session.query(Complaint)

            if search_filter:
                query = query.filter(
                    Complaint.complaint_text.ilike(f"%{search_filter}%") |
                    Complaint.category.ilike(f"%{search_filter}%")
                )

            if status_filter:
                query = query.filter_by(status=status_filter)

            complaints = query.order_by(Complaint.created_at.desc()).all()
            return [{
                "id": c.id,
                "customer_id": c.customer_id,
                "complaint_text": c.complaint_text,
                "category": c.category,
                "status": c.status,
                "created_at": c.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "customer":c.customer.to_dict() if c.customer else None
            } for c in complaints]
    except Exception as e:
        logging.error(f"Failed to get complaints: {str(e)}")
        return []


def update_complaint_status(complaint_id: int, new_status: str) -> bool:
    """Update complaint status and handle resolution timestamp"""
    if new_status not in [s.value for s in ComplaintStatus]:
        logging.error(f"Invalid status: {new_status}")
        return False

    try:
        with session_scope() as session:
            complaint = session.query(Complaint).get(complaint_id)
            if not complaint:
                logging.error(f"Complaint {complaint_id} not found")
                return False

            complaint.status = new_status

            if new_status == ComplaintStatus.RESOLVED.value:
                routing = session.query(Routing)\
                    .filter_by(complaint_id=complaint_id)\
                    .first()
                if routing:
                    routing.resolved_at = datetime.utcnow()

            return True
    except Exception as e:
        logging.error(f"Failed to update complaint status: {str(e)}")
        return False

def get_live_complaint_stats():
    try:
        with session_scope() as session:
            status_counts = session.query(
                Complaint.status,
                func.count(Complaint.id)
                ).group_by(Complaint.status).all()

            stats = {status.value: 0 for status in ComplaintStatus}
            stats.update({status:count for status, count in status_counts})
            return stats
    except Exception as e:
      logging.error(f"Error getting complaint statistics: {str(e)}")
      return {status.value: 0 for status in Complaintstatus}
    
def get_customer_by_account(account_number: str) -> Optional[Dict[str, Any]]:
    """Get customer by account number with decrypted data"""
    try:
        with session_scope() as session:
            encrypted_account = encrypt_data(account_number)
            customer = session.query(Customer)\
                .filter_by(account_number=encrypted_account)\
                .first()

            if not customer:
                return None

            return customer.to_dict(include_sensitive=True)
    except Exception as e:
        logging.error(f"Failed to get customer: {str(e)}")
        return None


def get_recent_complaints(hours: int = 24, limit: int = 100) -> List[Dict[str, Any]]:
    """Get recent complaints within specified hours"""
    try:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        with session_scope() as session:
            complaints = session.query(Complaint)\
                .filter(Complaint.created_at >= cutoff)\
                .order_by(Complaint.created_at.desc())\
                .limit(limit)\
                .all()
            return [c.to_dict() for c in complaints]
    except Exception as e:
        logging.error(f"Failed to get recent complaints: {str(e)}")
        return []


def delete_complaint(complaint_id: int) -> bool:
    """Delete a complaint and its associated routing"""
    try:
        with session_scope() as session:
            # Delete routing first to maintain referential integrity
            session.query(Routing)\
                .filter_by(complaint_id=complaint_id)\
                .delete()

            # Then delete the complaint
            session.query(Complaint)\
                .filter_by(id=complaint_id)\
                .delete()

            return True
    except Exception as e:
        logging.error(f"Failed to delete complaint: {str(e)}")
        return False