import requests

BASE_URL = "http://localhost:5000"

# Test 1: Health Check
def test_health():
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200 and response.json()["status"] == "ok":
        print("✅ Test 1 PASSED: Health check working")
    else:
        print("❌ Test 1 FAILED: Health check not working")

# Test 2: Predict with no data
def test_predict_bad_data():
    response = requests.post(f"{BASE_URL}/predict", json={"image": "baddata123"})
    if response.status_code == 400 or response.json()["hand_detected"] == False:
        print("✅ Test 2 PASSED: Bad data handled correctly")
    else:
        print("❌ Test 2 FAILED: Bad data not handled")

# Test 3: Predict with empty body
def test_predict_empty():
    response = requests.post(f"{BASE_URL}/predict", json={})
    if response.status_code == 400:
        print("✅ Test 3 PASSED: Empty request handled correctly")
    else:
        print("❌ Test 3 FAILED: Empty request not handled")

# Test 4: Speak with text
def test_speak():
    response = requests.post(f"{BASE_URL}/speak", json={"text": "Hello"})
    if response.status_code == 200:
        print("✅ Test 4 PASSED: Speak endpoint working")
    else:
        print("❌ Test 4 FAILED: Speak endpoint not working")

# Test 5: Speak with empty text
def test_speak_empty():
    response = requests.post(f"{BASE_URL}/speak", json={"text": ""})
    if response.status_code == 400:
        print("✅ Test 5 PASSED: Empty text handled correctly")
    else:
        print("❌ Test 5 FAILED: Empty text not handled")

# Run all tests
print("=============================")
print("   ISL API Test Report")
print("=============================")
test_health()
test_predict_bad_data()
test_predict_empty()
test_speak()
test_speak_empty()
print("=============================")
print("Testing Complete!")
print("=============================")
