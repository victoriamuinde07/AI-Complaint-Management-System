from datetime import datetime, timedelta
from functools import wraps
import base64
import logging
from typing import Dict, Any
from uuid import uuid4

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import jwt
from jwt.exceptions import InvalidTokenError

from models import (
    ComplaintStatus, add_customer, add_complaint, 
    get_all_complaints, get_customer_by_account, Customer,
    Complaint, Routing, decrypt_data
)
from database import init_db, db_session, get_encryption_service, Base

# Initialize Flask App
app = Flask(__name__)

# Configuration
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=1),
    JSONIFY_PRETTYPRINT_REGULAR=False,
    JSON_SORT_KEYS=False
)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', os.urandom(24))

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
    strategy="fixed-window"
)

# Initialize database
init_db()

class CryptoService:
    """Handles all cryptographic operations"""
    def __init__(self):
        self.ecdh_private = ec.generate_private_key(ec.SECP384R1())
        self.ed25519_private = ed25519.Ed25519PrivateKey.generate()
    
    @property
    def ecdh_public(self):
        return self.ecdh_private.public_key()
    
    @property
    def ed25519_public(self):
        return self.ed25519_private.public_key()
    
    def encrypt(self, data: str, client_pub_key: ec.EllipticCurvePublicKey) -> Dict[str, Any]:
        """ECDH + AES-GCM encryption"""
        try:
            shared_key = self.ecdh_private.exchange(ec.ECDH(), client_pub_key)
            derived_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b'hybrid_encryption'
            ).derive(shared_key)

            nonce = os.urandom(12)
            ciphertext = AESGCM(derived_key).encrypt(nonce, data.encode(), None)

            return {
                'ciphertext': base64.b64encode(ciphertext).decode(),
                'nonce': base64.b64encode(nonce).decode(),
                'server_public_key': self.ecdh_public.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ).decode()
            }
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise APIError("Encryption failed", 500)

    def sign(self, data: str) -> Dict[str, Any]:
        """Ed25519 signature"""
        try:
            signature = self.ed25519_private.sign(data.encode())
            return {
                'signature': base64.b64encode(signature).decode(),
                'public_key': base64.b64encode(self.ed25519_public.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw
                )).decode()
            }
        except Exception as e:
            logger.error(f"Signing failed: {str(e)}")
            raise APIError("Signing failed", 500)

class APIError(Exception):
    """Custom API error class"""
    def __init__(self, message, status_code=400, payload=None):
        super().__init__()
        self.message = message
        self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

# Initialize services
crypto_service = CryptoService()
encryption = get_encryption_service()

# JWT Configuration
JWT_ALGORITHM = 'ES256'
JWT_EXP_DELTA = timedelta(hours=1)

# Department mapping
department_map = {
    "credit_card": "Card Services",
    "retail_banking": "Customer Support",
    "credit_reporting": "Credit Department",
    "mortgages_and_loans": "Loans Department",
    "debt_collection": "Collections"
}

# Middleware
@app.before_request
def before_request():
    """Assign request ID and log incoming requests"""
    request.request_id = uuid4().hex
    logger.info(f"Request {request.request_id} started: {request.method} {request.path}")

@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    headers = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'Content-Security-Policy': "default-src 'self'",
        'Strict-Transport-Security': 'max-age=63072000; includeSubDomains; preload',
        'X-XSS-Protection': '1; mode=block',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
        'Server': 'RiaorSystem',
        'Cache-Control': 'no-store, max-age=0'
    }
    response.headers.update(headers)
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    """Global error handler"""
    if isinstance(e, APIError):
        return jsonify(e.to_dict()), e.status_code
    elif isinstance(e, HTTPException):
        return jsonify({"message": e.description}), e.code
    
    logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({"message": "Internal server error"}), 500

# Helper functions
def validate_input(data: dict, required_fields: list) -> bool:
    """Validate request payload"""
    if not data:
        raise APIError("No data provided", 400)

    missing = [field for field in required_fields if field not in data]
    if missing:
        raise APIError(f"Missing fields: {', '.join(missing)}", 400)
    return True

def validate_account_number(account_number: str) -> bool:
    """Validate account number format"""
    return account_number.isdigit() and 5 <= len(account_number) <= 20

def generate_jwt_token(user_id: str, is_admin: bool = False) -> str:
    """Generate JWT token"""
    payload = {
        'user_id': user_id,
        'is_admin': is_admin,
        'exp': datetime.utcnow() + JWT_EXP_DELTA,
        'jti': uuid4().hex
    }
    return jwt.encode(payload, crypto_service.ecdh_private, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> dict:
   
    try:
        return jwt.decode(token, crypto_service.ecdh_public, algorithms=[JWT_ALGORITHM])
    except InvalidTokenError as e:
        raise APIError("Invalid token", 401) from e

def log_access(user_id: str, action: str, details: str):
    """Audit logging"""
    logger.info(f"[AUDIT] {user_id} {action} | {details}")

# Decorators
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            raise APIError("Authorization token missing", 401)

        token = auth_header.split(' ')[1]
        try:
            payload = verify_jwt_token(token)
            request.user = payload
        except APIError:
            raise
        except Exception as e:
            logger.error(f"Token validation error: {str(e)}")
            raise APIError("Invalid token", 401)

        return f(*args, **kwargs)
    return decorated



def admin_required(f):
    @wraps(f)
    @token_required
    def decorated(*args, **kwargs):
        if not request.user.get('is_admin'):
            raise APIError("Admin access required", 403)
        return f(*args, **kwargs)
    return decorated

# Routes
@app.route("/api/public-key", methods=["GET"])
def get_public_key():
    """Get server's public keys for encryption"""
    return jsonify({
        "ecdh_public_key": crypto_service.ecdh_public.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode(),
        "ed25519_public_key": base64.b64encode(
            crypto_service.ed25519_public.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
        ).decode()
    })

@app.route("/debug/complaints")
def debug_complaints():
    with SessionLocal() as session:
        complaints = session.query(Complaint).all()
        return jsonify([{
            'id': c.id,
            'customer_id': c.customer_id,
            'text': c.complaint_text[:50] + '...' if c.complaint_text else None,
            'category': c.category,
            'status': c.status,
            'created_at': c.created_at.isoformat() if c.created_at else None
        } for c in complaints])

@app.route("/api/version", methods=["GET"])
def version_info():
    """API version information"""
    return jsonify({
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "features": ["encryption", "signing", "rate_limiting"]
    })

@app.route("/api/auth/login", methods=["POST"])
@limiter.limit("5/minute")
def login():
    """User authentication endpoint"""
    try:
        data = request.get_json()
        validate_input(data, ["username", "password"])

        # TODO: Replace with real authentication
        if data["username"] == "admin" and data["password"] == os.getenv("ADMIN_PASSWORD"):
            token = generate_jwt_token("admin", True)
            return jsonify({"token": token}), 200

        raise APIError("Invalid credentials", 401)
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise APIError("Login failed", 500)

@app.route("/api/customers", methods=["POST"])
@limiter.limit("10/minute")
def add_customer_endpoint():
    """Create new customer"""
    try:
        data = request.get_json()
        validate_input(data, ["name", "phone_number", "account_number"])

        if not validate_account_number(data["account_number"]):
            raise APIError("Invalid account number format", 400)

        customer_id = add_customer(
            name=data["name"],
            phone_number=data["phone_number"],
            account_number=data["account_number"]
        )

        if not customer_id:
            raise APIError("Customer creation failed", 400)

        return jsonify({
            "message": "Customer created",
            "customer_id": customer_id
        }), 201
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Customer creation error: {str(e)}", exc_info=True)
        raise APIError("Internal server error", 500)

@app.route("/api/complaints", methods=["POST"])
@token_required
@limiter.limit("10/minute")
def add_complaint_endpoint():
    """Submit new complaint"""
    try:
        data = request.get_json()
        validate_input(data, ["account_number", "complaint_text", "category"])

        if len(data["complaint_text"]) < 10:
            raise APIError("Complaint text too short", 400)

        with db_session() as session:
            # Find customer by decrypted account number
            customers = session.query(Customer).all()
            customer = next(
                (c for c in customers 
                 if decrypt_data(c.account_number) == data["account_number"]),
                None
            )
            
            if not customer:
                raise APIError("Customer not found", 404)

            new_complaint = Complaint(
                customer_id=customer.id,
                complaint_text=data["complaint_text"],
                category=data["category"],
                status=ComplaintStatus.SUBMITTED.value
            )
            session.add(new_complaint)
            session.flush()

            routing = Routing(
                customer_id=customer.id,
                complaint_id=new_complaint.id,
                customer_account_number=customer.account_number,
                account_display=customer.account_display,
                route_name=department_map.get(data["category"].lower(), "General Support")
            )
            session.add(routing)
            
            return jsonify({
                "complaint_id": new_complaint.id,
                "status": "submitted"
            }), 201
    except Exception as e:
        logger.error(f"Complaint submission error: {str(e)}", exc_info=True)
        raise APIError("Submission failed", 500)

@app.route("/api/complaints/stats", methods=["GET"])
@token_required
def get_complaint_stats():
    """Get complaint statistics"""
    try:
        from models import get_live_complaint_stats
        stats = get_live_complaint_stats()
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"Failed to get complaint stats: {str(e)}", exc_info=True)
        raise APIError("Failed to load statistics", 500)

@app.route("/api/complaints", methods=["GET"])
@token_required
def get_complaints_endpoint():
    """Get all complaints"""
    try:
        complaints = get_all_complaints()
        return jsonify([{
            "id": c['id'],
            "category": c['category'],
            "status": c['status'],
            "created_at": c['created_at'],
            "customer_id": c['customer_id']
        } for c in complaints]), 200
    except Exception as e:
        logger.error(f"Complaint retrieval error: {str(e)}", exc_info=True)
        raise APIError("Internal server error", 500)

@app.route("/api/customers/<account_number>", methods=["GET"])
@token_required
def get_customer_endpoint(account_number):
    """Get customer details"""
    try:
        if not validate_account_number(account_number):
            raise APIError("Invalid account number format", 400)

        customer = get_customer_by_account(account_number)
        if not customer:
            raise APIError("Customer not found", 404)
            
        include_sensitive = request.user.get('is_admin', False)
        return jsonify(customer.to_dict(include_sensitive=include_sensitive)), 200
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Customer retrieval error: {str(e)}", exc_info=True)
        raise APIError("Internal server error", 500)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    })

if __name__ == "__main__":
    # Validate configuration
    required_env_vars = ["FLASK_SECRET_KEY", "ENCRYPTION_KEY"]
    missing = [var for var in required_env_vars if not os.getenv(var)]
    if missing:
        logger.critical(f"Missing required environment variables: {missing}")
        raise RuntimeError("Configuration incomplete")

    # Start application
    ssl_enabled = os.getenv("SSL_ENABLED", "false").lower() == "true"
    ssl_context = ('cert.pem', 'key.pem') if ssl_enabled else None

    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "5000")),
        ssl_context=ssl_context,
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true"
    )