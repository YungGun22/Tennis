import argparse
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="Tennis Match Exact Score Prediction Engine")
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    subparsers.add_parser("ingest", help="Ingest ATP/WTA match data")
    subparsers.add_parser("clean", help="Clean and normalize dataset")
    subparsers.add_parser("build_elo", help="Compute dynamic Elo ratings")
    subparsers.add_parser("build_features", help="Extract chronological leak-free match features")
    subparsers.add_parser("train_set_model", help="Train Base Set-Win Model")
    subparsers.add_parser("simulate_match", help="Simulate exact score probabilities for a match")
    subparsers.add_parser("run_backtest", help="Run walk-forward evaluation and betting backtest")

    # API Server Command
    api_parser = subparsers.add_parser("serve", help="Launch FastAPI REST API server")
    api_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    api_parser.add_argument("--port", type=int, default=8000, help="Port number")

    args = parser.parse_args()

    if args.command == "serve":
        print(f"[STAGE 9] Starting Tennis Prediction REST API Server on {args.host}:{args.port}...")
        uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=True)

if __name__ == "__main__":
    main()