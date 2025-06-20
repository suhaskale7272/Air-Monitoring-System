const express = require('express');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const cors = require('cors');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;

const config = {
    channelId: '2918627',
    apiKey: '3TW5RRLGEWUWSPES',
    dataDir: path.join(__dirname, 'data'),
    dataFile: 'sensor_readings.csv',
    modelScript: 'air_quality_ml.py',
    updateInterval: 5 * 60 * 1000, // 5 minutes
    maxDataPoints: 1000 // Limit number of points to prevent front-end overload
};

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

if (!fs.existsSync(config.dataDir)) {
    fs.mkdirSync(config.dataDir, { recursive: true });
}

// Enhanced data fetching with PM2.5 support
async function fetchThingSpeakData() {
    try {
        const response = await axios.get(
            `https://api.thingspeak.com/channels/${config.channelId}/feeds.json`,
            {
                params: {
                    api_key: config.apiKey,
                    results: 8000
                },
                timeout: 10000
            }
        );

        if (!response.data?.feeds) {
            throw new Error('Invalid data structure from ThingSpeak');
        }

        return response.data.feeds
            .map(feed => ({
                timestamp: feed.created_at,
                temperature: parseFloat(feed.field1),
                humidity: parseFloat(feed.field2),
                gas: parseFloat(feed.field3),
                pm25: parseFloat(feed.field4) || null // Handle missing PM2.5 data
            }))
            .filter(data => (
                !isNaN(data.temperature) && 
                !isNaN(data.humidity) && 
                !isNaN(data.gas) &&
                (data.pm25 === null || !isNaN(data.pm25)) &&
                data.temperature >= -20 && data.temperature <= 60 &&
                data.humidity >= 0 && data.humidity <= 100 &&
                data.gas >= 0 &&
                (data.pm25 === null || data.pm25 >= 0)
            );
    } catch (error) {
        console.error('Fetch error:', error.message);
        return null;
    }
}

async function updateDataFile() {
    const feeds = await fetchThingSpeakData();
    if (!feeds || feeds.length === 0) {
        throw new Error('No valid data available');
    }

    // Limit number of data points to prevent front-end overload
    const limitedFeeds = feeds.slice(-config.maxDataPoints);

    const csvData = ['timestamp,temperature,humidity,gas,pm25'];
    limitedFeeds.forEach(feed => {
        csvData.push(`${feed.timestamp},${feed.temperature},${feed.humidity},${feed.gas},${feed.pm25 || ''}`);
    });

    const filePath = path.join(config.dataDir, config.dataFile);
    fs.writeFileSync(filePath, csvData.join('\n'));
    console.log(`Updated data file with ${limitedFeeds.length} records`);
    return true;
}

function runPythonModel(args = []) {
    return new Promise((resolve, reject) => {
        const scriptPath = path.join(__dirname, config.modelScript);
        const process = spawn('python', [scriptPath, ...args], {
            cwd: __dirname,
            stdio: ['pipe', 'pipe', 'pipe']
        });

        let stdout = '';
        let stderr = '';
        let timeout;

        process.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        process.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        process.on('error', (error) => {
            clearTimeout(timeout);
            reject(new Error(`Process error: ${error.message}`));
        });

        process.on('close', (code) => {
            clearTimeout(timeout);
            if (code !== 0 || stderr) {
                reject(new Error(`Model failed: ${stderr || `Exit code ${code}`}`));
            } else {
                try {
                    const output = JSON.parse(stdout.trim());
                    if (output.error) {
                        reject(new Error(output.error));
                    } else {
                        resolve(output);
                    }
                } catch (e) {
                    reject(new Error('Invalid model output format'));
                }
            }
        });

        timeout = setTimeout(() => {
            process.kill();
            reject(new Error('Model execution timed out'));
        }, 30000);
    });
}

// New endpoint for latest readings
app.get('/api/latest', async (req, res) => {
    try {
        const filePath = path.join(config.dataDir, config.dataFile);
        if (!fs.existsSync(filePath)) {
            await updateDataFile();
        }

        const csvData = fs.readFileSync(filePath, 'utf8');
        const lines = csvData.split('\n').filter(line => line.trim());
        
        if (lines.length <= 1) {
            throw new Error('No data available');
        }

        const lastLine = lines[lines.length - 1];
        const [timestamp, temp, hum, gas, pm25] = lastLine.split(',');

        const latestReading = {
            timestamp,
            temperature: parseFloat(temp),
            humidity: parseFloat(hum),
            gas: parseFloat(gas),
            pm25: pm25 ? parseFloat(pm25) : null
        };

        res.json({ success: true, data: latestReading });
    } catch (error) {
        console.error('Latest endpoint error:', error.message);
        res.status(500).json({
            success: false,
            error: 'Failed to get latest reading',
            details: error.message
        });
    }
});

app.get('/api/data', async (req, res) => {
    try {
        const filePath = path.join(config.dataDir, config.dataFile);
        if (!fs.existsSync(filePath)) {
            await updateDataFile();
        }

        const csvData = fs.readFileSync(filePath, 'utf8');
        const lines = csvData.split('\n').slice(1).filter(line => line);
        
        // Sample data if we have too many points
        const sampleStep = Math.max(1, Math.floor(lines.length / 200));
        const sampledLines = lines.filter((_, index) => index % sampleStep === 0);

        const feeds = sampledLines.map(line => {
            const [timestamp, temp, hum, gas, pm25] = line.split(',');
            return {
                timestamp,
                temperature: parseFloat(temp),
                humidity: parseFloat(hum),
                gas: parseFloat(gas),
                pm25: pm25 ? parseFloat(pm25) : null
            };
        }).filter(f => 
            f.timestamp && 
            !isNaN(f.temperature) && 
            !isNaN(f.humidity) && 
            !isNaN(f.gas) &&
            (f.pm25 === null || !isNaN(f.pm25))
        );

        res.json({ success: true, data: feeds });
    } catch (error) {
        console.error('Data endpoint error:', error.message);
        res.status(500).json({
            success: false,
            error: 'Data retrieval failed',
            details: error.message
        });
    }
});

// Enhanced prediction endpoint with PM2.5 support
app.get('/api/predict', async (req, res) => {
    try {
        const temp = parseFloat(req.query.temperature);
        const hum = parseFloat(req.query.humidity);
        const pm25 = req.query.pm25 ? parseFloat(req.query.pm25) : null;
        
        if (isNaN(temp) || isNaN(hum)) {
            return res.status(400).json({
                success: false,
                error: 'Invalid parameters',
                details: 'Temperature and humidity must be numbers'
            });
        }

        const inputData = `created_at,field1,field2,field4\n"${new Date().toISOString()}",${temp},${hum},${pm25 || ''}`;
        const inputFile = path.join(config.dataDir, 'predict_input.csv');
        fs.writeFileSync(inputFile, inputData);

        const result = await runPythonModel([
            '--predict', inputFile,
            '--data', path.join(config.dataDir, config.dataFile)
        ]);

        // Enhanced response with PM2.5 quality assessment
        res.json({
            success: true,
            prediction: {
                gasLevel: result.gas_level,
                gasType: result.gas_type,
                airQuality: result.air_quality,
                pm25Quality: pm25 !== null ? getPM25Quality(pm25) : null,
                confidence: result.confidence,
                timestamp: new Date().toISOString()
            }
        });
    } catch (error) {
        console.error('Prediction error:', error.message);
        res.status(500).json({
            success: false,
            error: 'Prediction failed',
            details: error.message
        });
    }
});

// Helper function for PM2.5 quality assessment
function getPM25Quality(pm25) {
    if (pm25 <= 12) return { label: "Good", class: "status-good" };
    if (pm25 <= 35) return { label: "Moderate", class: "status-moderate" };
    if (pm25 <= 55) return { label: "Unhealthy for Sensitive", class: "status-poor" };
    if (pm25 <= 150) return { label: "Unhealthy", class: "status-unhealthy" };
    return { label: "Hazardous", class: "status-hazardous" };
}

// New endpoint for location data
app.get('/api/location', (req, res) => {
    // In a real application, you might get this from the device or user input
    res.json({
        success: true,
        location: "IoT Device Location",
        coordinates: {
            lat: 37.7749,
            lng: -122.4194
        }
    });
});

// Endpoint to download CSV data
app.get('/api/download', (req, res) => {
    try {
        const filePath = path.join(config.dataDir, config.dataFile);
        if (!fs.existsSync(filePath)) {
            return res.status(404).json({
                success: false,
                error: 'Data file not found'
            });
        }

        res.download(filePath, 'air_quality_data.csv');
    } catch (error) {
        console.error('Download error:', error.message);
        res.status(500).json({
            success: false,
            error: 'Download failed',
            details: error.message
        });
    }
});

async function initialize() {
    try {
        await updateDataFile();
        setInterval(async () => {
            try {
                await updateDataFile();
            } catch (error) {
                console.error('Periodic update failed:', error.message);
            }
        }, config.updateInterval);

        app.listen(PORT, () => {
            console.log(`Server running on port ${PORT}`);
            console.log(`Data updates every ${config.updateInterval/60000} minutes`);
        });
    } catch (error) {
        console.error('Initialization failed:', error.message);
        process.exit(1);
    }
}

initialize();
