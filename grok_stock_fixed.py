def main():
    root = tk.Tk()
    
    try:
        app = GridStockGrokGUI(root)  # Use the grid-enabled version
        
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print("\nReceived shutdown signal")
            app.shutdown()
        
        # Register signal handlers
        if sys.platform != 'win32':  # Not on Windows
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGHUP, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        root.mainloop()
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        if 'app' in locals():
            app.shutdown()
        else:
            root.destroy()
        sys.exit(1)

if __name__ == "__main__":
    main() 