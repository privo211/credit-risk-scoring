#!/bin/bash
# Test script for the Credit Risk Scoring API
# Usage: ./scripts/test_api.sh [host]
# Default host: http://localhost:8000

HOST="${1:-http://localhost:8000}"
PASS=0
FAIL=0

green() { printf "\033[32m%s\033[0m\n" "$1"; }
red()   { printf "\033[31m%s\033[0m\n" "$1"; }

echo "=========================================="
echo "  Credit Risk Scoring API Tests"
echo "  Host: $HOST"
echo "=========================================="

# Test 1: Health endpoint
echo -n "Test 1: GET /health ... "
RESP=$(curl -s "$HOST/health")
if echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='healthy'; assert 'model_loaded' in d; print('OK')" 2>/dev/null; then
    green "PASS"
    ((PASS++))
else
    red "FAIL"
    echo "  Response: $RESP"
    ((FAIL++))
fi

# Test 2: Predict with full applicant
echo -n "Test 2: POST /predict (full data) ... "
RESP=$(curl -s -X POST "$HOST/predict" \
  -H "Content-Type: application/json" \
  -d @scripts/sample_payload.json)
if echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'probability' in d; assert 'risk_band' in d; assert 'model_version' in d; assert 0 <= d['probability'] <= 1; print('OK')" 2>/dev/null; then
    green "PASS"
    ((PASS++))
else
    red "FAIL"
    echo "  Response: $RESP"
    ((FAIL++))
fi

# Test 3: Predict with minimal applicant
echo -n "Test 3: POST /predict (minimal data) ... "
RESP=$(curl -s -X POST "$HOST/predict" \
  -H "Content-Type: application/json" \
  -d '{"duration": 12, "credit_amount": 2000, "age": 28}')
if echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'probability' in d; assert 'risk_band' in d; print('OK')" 2>/dev/null; then
    green "PASS"
    ((PASS++))
else
    red "FAIL"
    echo "  Response: $RESP"
    ((FAIL++))
fi

# Test 4: Batch predict
echo -n "Test 4: POST /batch_predict ... "
RESP=$(curl -s -X POST "$HOST/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{"applicants": [{"duration": 24, "credit_amount": 5000, "age": 35}, {"duration": 12, "credit_amount": 2000, "age": 28}]}')
if echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['count'] == 2; assert len(d['predictions']) == 2; print('OK')" 2>/dev/null; then
    green "PASS"
    ((PASS++))
else
    red "FAIL"
    echo "  Response: $RESP"
    ((FAIL++))
fi

# Test 5: Invalid input returns 422
echo -n "Test 5: POST /predict (invalid data -> 422) ... "
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$HOST/predict" \
  -H "Content-Type: application/json" \
  -d '{"foo": "bar"}')
if [ "$HTTP_CODE" -eq 422 ]; then
    green "PASS"
    ((PASS++))
else
    red "FAIL (HTTP $HTTP_CODE)"
    ((FAIL++))
fi

echo "=========================================="
echo "  Results: $PASS passed, $FAIL failed"
echo "=========================================="
exit $FAIL
