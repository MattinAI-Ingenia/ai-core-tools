from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship
from db.database import Base
from datetime import datetime


class RefreshToken(Base):
    """Persisted refresh-token record for local-auth rotation and reuse detection.

    The raw token is never stored. `token_hash` holds the SHA-256 hex digest of
    the opaque token value. `rotated_at` is set when this jti has been exchanged
    for a new token; a non-null `rotated_at` on an incoming token signals theft.
    """
    __tablename__ = 'refresh_tokens'

    id = Column(Integer, primary_key=True)
    jti = Column(String(64), nullable=False, unique=True, index=True)
    user_id = Column(Integer, ForeignKey('User.user_id', ondelete='CASCADE'), nullable=False, index=True)
    family_id = Column(String(64), nullable=False, index=True)
    token_hash = Column(String(128), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    rotated_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    user_agent = Column(String(255), nullable=True)
    ip = Column(String(45), nullable=True)

    # Relationships
    user = relationship('User', back_populates='refresh_tokens')
