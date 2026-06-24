"""
test_auth.py — authentication tests.

Two layers:
  * Unit tests for the security helpers (hashing, JWT) — no app or DB needed.
  * Integration tests for /auth/* against an in-memory SQLite database.

Important: we mount ONLY the auth router on a throwaway FastAPI app. That keeps
this module completely independent of app.main / app.inference, so it never
imports torch, never loads the ML model, and never shares dependency overrides
with the other test files.
"""
import os

os.environ.setdefault("DB_USERNAME", "testuser")
os.environ.setdefault("DB_PASSWORD", "testpass")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")
os.environ.setdefault("DB_NAME", "testdb")

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.models import User
from app.routers import auth as auth_router
from app.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

# ---- In-memory SQLite shared across connections for the whole module ----
test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)
Base.metadata.create_all(bind=test_engine)


def _override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# A minimal app with only the auth routes — no model, no lifespan.
test_app = FastAPI()
test_app.include_router(auth_router.router)
test_app.dependency_overrides[get_db] = _override_get_db


@pytest.fixture
def client():
    return TestClient(test_app)


@pytest.fixture(autouse=True)
def clean_users():
    """Start every test with an empty users table."""
    db = TestingSessionLocal()
    db.execute(delete(User))
    db.commit()
    db.close()
    yield


# ---- Helpers ----

def _register(client, email="alice@example.com", password="supersecret", full_name="Alice"):
    return client.post(
        "/auth/register",
        json={"email": email, "password": password, "full_name": full_name},
    )


def _login(client, email="alice@example.com", password="supersecret"):
    return client.post("/auth/login", json={"email": email, "password": password})


# ---- Unit tests: security helpers ----

class TestSecurityHelpers:

    def test_hash_is_not_plaintext(self):
        assert hash_password("supersecret") != "supersecret"

    def test_verify_accepts_correct_password(self):
        hashed = hash_password("supersecret")
        assert verify_password("supersecret", hashed) is True

    def test_verify_rejects_wrong_password(self):
        hashed = hash_password("supersecret")
        assert verify_password("wrongpass", hashed) is False

    def test_each_hash_uses_a_fresh_salt(self):
        assert hash_password("supersecret") != hash_password("supersecret")

    def test_token_roundtrip_returns_subject(self):
        token = create_access_token(subject="alice@example.com")
        assert decode_access_token(token) == "alice@example.com"

    def test_decode_invalid_token_returns_none(self):
        assert decode_access_token("not-a-real-token") is None


# ---- Integration tests: /auth/register ----

class TestRegister:

    def test_register_returns_201(self, client):
        assert _register(client).status_code == 201

    def test_register_returns_user_payload(self, client):
        data = _register(client).json()
        assert data["email"] == "alice@example.com"
        assert data["full_name"] == "Alice"
        assert "id" in data

    def test_register_never_returns_password(self, client):
        data = _register(client).json()
        assert "password" not in data
        assert "hashed_password" not in data

    def test_register_duplicate_email_returns_409(self, client):
        _register(client)
        assert _register(client).status_code == 409

    def test_register_short_password_returns_422(self, client):
        response = client.post(
            "/auth/register", json={"email": "bob@example.com", "password": "short"}
        )
        assert response.status_code == 422

    def test_register_invalid_email_returns_422(self, client):
        response = client.post(
            "/auth/register", json={"email": "not-an-email", "password": "supersecret"}
        )
        assert response.status_code == 422


# ---- Integration tests: /auth/login ----

class TestLogin:

    def test_login_returns_access_token(self, client):
        _register(client)
        data = _login(client).json()
        assert "access_token" in data and data["access_token"]

    def test_login_token_type_is_bearer(self, client):
        _register(client)
        assert _login(client).json()["token_type"] == "bearer"

    def test_login_includes_user(self, client):
        _register(client)
        assert _login(client).json()["user"]["email"] == "alice@example.com"

    def test_login_wrong_password_returns_401(self, client):
        _register(client)
        assert _login(client, password="incorrect").status_code == 401

    def test_login_unknown_email_returns_401(self, client):
        assert _login(client, email="ghost@example.com").status_code == 401


# ---- Integration tests: protected routes ----

class TestProtectedRoutes:

    def test_me_with_token_returns_current_user(self, client):
        _register(client)
        token = _login(client).json()["access_token"]
        response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json()["email"] == "alice@example.com"

    def test_me_without_token_returns_401(self, client):
        # /auth/me uses the same dependency that guards /predict.
        assert client.get("/auth/me").status_code == 401

    def test_me_with_garbage_token_returns_401(self, client):
        response = client.get("/auth/me", headers={"Authorization": "Bearer nonsense"})
        assert response.status_code == 401
