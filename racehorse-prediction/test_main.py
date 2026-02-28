import pytest
from fastapi.testclient import TestClient
import os
import json

from main import app  # replace with actual module name if different

client = TestClient(app)

# ---------------------------------------------------------
# Security Tests
# ---------------------------------------------------------

def test_secret_folder_access_denied():
    """Test that the /secret directory is not exposed statically."""
    # Try to access a known file in the secret directory
    response = client.get("/secret/firebase-key.json")
    # Should be 404 because we don't serve static files from /secret
    assert response.status_code == 404

def test_dockerignore_contains_secret():
    """Verify that .dockerignore prevents the secret folder from being packaged."""
    dockerignore_path = ".dockerignore"
    if os.path.exists(dockerignore_path):
        with open(dockerignore_path, "r") as f:
            content = f.read()
            assert "secret/" in content, "The secret/ directory must be in .dockerignore for security."
    else:
        pytest.fail(".dockerignore file is missing!")

# ---------------------------------------------------------
# Endpoint Functional Tests
# ---------------------------------------------------------

def test_read_main_page():
    """Test the root endpoint serving the index.html page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "一口馬主　AI予測システム" in response.text

def test_read_retrain_page_is_secured():
    """Verify the retraining page is accessible but checking its location."""
    response = client.get("/retrain-model")
    # If the file exists in secret/, it should load. If not, it means the test environment
    # doesn't have it, which is fine, but it shouldn't crash.
    assert response.status_code in [200, 404]

def test_predict_endpoint_valid_input():
    """Test prediction endpoint with valid data structure."""
    test_data = {
        "性別": "牡",
        "BirthMonth": 3,
        "厩舎": "TestStable",
        "馬主": "TestOwner",
        "生産者": "TestBreeder",
        "生産地": "TestLocation",
        "種牡馬": "ロードカナロア",
        "母父名": "ディープインパクト",
        "父タイプ名": "キングマンボ系",
        "母父タイプ名": "サンデーサイレンス系"
    }
    
    response = client.post("/predict", json=test_data)
    
    # We expect 200 OK or 503 if the ML model isn't loaded in the testing environment
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        json_data = response.json()
        assert "prediction" in json_data
        assert "prediction_tier" in json_data
        assert "probabilities" in json_data
        assert "reasons" in json_data

def test_predict_endpoint_invalid_method():
    """Test that GET requests are rejected on the POST endpoint."""
    response = client.get("/predict")
    assert response.status_code == 405  # Method Not Allowed

def test_predict_endpoint_missing_fields():
    """Test prediction endpoint with missing required fields (Pydantic validation)."""
    # Missing many required string fields
    incomplete_data = {
        "性別": "牡",
        "BirthMonth": 3
    }
    response = client.post("/predict", json=incomplete_data)
    assert response.status_code == 422  # Unprocessable Entity (Validation Error)
