# Test Report — ISL Sign Language Recognition
**Project:** Real-Time Indian Sign Language Recognition and Speech Conversion
**Tester:** Ananya Jantli
**College:** KLE Technological University, 2025-26

---

## Module 1: Data Collection (Riya Kamat)
- [ ] landmarks.csv exists
- [ ] Correct columns present
- [ ] 200+ rows per gesture class
- [ ] No empty/blank cells

**Status: Pending**
**Date Tested:** —
**Notes:** —

---

## Module 2: Backend API (Sonali G)
- [x] GET /health returns {status: ok}
- [x] POST /predict with bad data handled correctly
- [x] POST /predict with empty body returns 400
- [x] POST /speak with text returns 200
- [x] POST /speak with empty text returns 400

**Status: ✅ ALL 5 TESTS PASSED**
**Date Tested: 01/05/2026**
**Notes: All API endpoints working correctly. Server running on port 5000.**

---

## Module 3: Frontend (Tanushree H)
- [ ] Webcam turns on when page loads
- [ ] Browser asks for webcam permission
- [ ] Prediction updates when hand is shown
- [ ] Speak button produces audio
- [ ] Clear button resets everything
- [ ] Guide page shows all gestures
- [ ] Search bar works
- [ ] Tested on Chrome
- [ ] Tested on Firefox
- [ ] Tested on Edge

**Status: Pending**
**Date Tested:** —
**Notes:** —

---

## Integration Test
- [ ] Backend starts without errors
- [ ] Frontend opens in browser
- [ ] Letter A shown → A appears on screen
- [ ] 5 gestures in a row → text builds correctly
- [ ] Speak button plays audio
- [ ] Clear button resets
- [ ] Works in dark room
- [ ] Works with different person's hand

**Status: Pending**
**Date Tested:** —
**Notes:** —
