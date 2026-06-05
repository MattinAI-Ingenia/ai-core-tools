from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import relationship
from db.database import Base
from datetime import datetime

class User(Base):
    '''User model class constructor'''
    __tablename__ = 'User'
    user_id = Column(Integer, primary_key=True)
    email = Column(String(255))
    name = Column(String(255))
    create_date = Column(DateTime, default=datetime.now)
    is_active = Column(Boolean, default=True, nullable=False)
    auth_method = Column(String(50), default='oidc', nullable=True)
    email_verified = Column(Boolean, default=True, nullable=False)

    # Relationships
    # Deletion taxonomy (see spec user-deletion-and-app-transfer, AD-1):
    #  - owned_apps / api_keys / app_collaborations are Class-B/C: emptied explicitly
    #    by UserService.delete_user BEFORE db.delete(user). passive_deletes=True stops
    #    the ORM from emitting an UPDATE ... SET <fk>=NULL on a delete (those FKs are
    #    NO ACTION + NOT NULL and would raise); the orchestration leaves these empty.
    #  - subscription is Class-A: DB-level ON DELETE CASCADE handles it; passive_deletes
    #    lets the DB cascade run instead of an ORM SET NULL that fights it.
    owned_apps = relationship('App', foreign_keys='App.owner_id', back_populates='owner', lazy=True,
                              passive_deletes=True)
    app_collaborations = relationship('AppCollaborator', foreign_keys='AppCollaborator.user_id', back_populates='user', lazy=True,
                                      passive_deletes=True)
    api_keys = relationship('APIKey', back_populates='user', lazy=True,
                            passive_deletes=True)
    subscription = relationship('Subscription', back_populates='user', uselist=False, lazy=True,
                                cascade='all, delete-orphan', passive_deletes=True)
    credential = relationship('UserCredential', back_populates='user', uselist=False, lazy=True,
                              cascade='all, delete-orphan', passive_deletes=True)
    refresh_tokens = relationship('RefreshToken', back_populates='user', lazy=True,
                                  cascade='all, delete-orphan', passive_deletes=True)

    def get_id(self):
        return self.user_id
    
    @property
    def apps(self):
        """Get all apps user has access to (owned + collaborated)"""
        from services.user_service import UserService
        return UserService.get_user_accessible_apps(self.user_id) 