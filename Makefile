# Realitic — quick commands from repo root

.PHONY: run run-flutter run-backend

# Start Flutter app (web, Chrome)
run-flutter:
	cd frontend && flutter run -d chrome

# Start backend API
run-backend:
	uvicorn app.main:app --reload

# Alias: "make run" starts Flutter
run: run-flutter
