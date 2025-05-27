# Copy the service file to the systemd directory for custom services
sudo cp triton_manager.service /etc/systemd/system/triton_manager.service

# Reload the systemd manager configuration to recognize the new service file
sudo systemctl daemon-reload

# Start the triton_manager service immediately
sudo systemctl start triton_manager.service

# Enable the service to start automatically at boot
sudo systemctl enable triton_manager.service

# Check the current status of the triton_manager service to ensure it’s running correctly
sudo systemctl status triton_manager.service

# View real-time logs for the triton_manager service
journalctl -u triton_manager.service -f
