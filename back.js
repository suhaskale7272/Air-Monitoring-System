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
    updateInterval: 5 * 60 * 1000
};

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

if (!fs.existsSync(config.dataDir)) {
    fs.mkdirSync(config.dataDir, { recursive: true });
}

async function fetchThingSpeakData() {
    try {
        // Request field1 (temperature), field2 (humidity), field3 (gas), field4 (PM2.5)
        const response = await axios.get(
            https://api.thingspeak.com/channels/${config.channelId}/feeds.json,
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

        return response.data.feeds.filter(feed => {
            try {
                const temp = parseFloat(feed.field1);
                const hum = parseFloat(feed.field2);
                const gas = parseFloat(feed.field3);
                const pm25 = parseFloat(feed.field4); // Assuming field4 for PM2.5
                return !isNaN(temp) && !isNaN(hum) && !isNaN(gas) && !isNaN(pm25) &&
                       temp >= -20 && temp <= 60 &&
                       hum >= 0 && hum <= 100 &&
                       gas >= 0 && pm25 >= 0; // Validate PM2.5 as well
            } catch (e) {
                return false;
            }
        });
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

    // Include 'pm25' in the header
    const csvData = ['timestamp,temperature,humidity,gas,pm25'];
    feeds.forEach(feed => {
        csvData.push(${feed.created_at},${feed.field1},${feed.field2},${feed.field3},${feed.field4}); // Include field4 for PM2.5
    });

    const filePath = path.join(config.dataDir, config.dataFile);
    fs.writeFileSync(filePath, csvData.join('\n'));
    console.log(Updated data file with ${feeds.length} records);
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
            reject(new Error(Process error: ${error.message}));
        });

        process.on('close', (code) => {
            clearTimeout(timeout);
            if (code !== 0 || stderr) {
                reject(new Error(`Model failed: ${stderr || Exit code ${code}}`));
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

app.get('/api/data', async (req, res) => {
    try {
        const filePath = path.join(config.dataDir, config.dataFile);
        if (!fs.existsSync(filePath)) {
            await updateDataFile();
        }

        const csvData = fs.readFileSync(filePath, 'utf8');
        const lines = csvData.split('\n').slice(1).filter(line => line);
        const feeds = lines.map(line => {
            const [timestamp, temp, hum, gas, pm25] = line.split(','); // Destructure with pm25
            return {
                timestamp,
                temperature: parseFloat(temp),
                humidity: parseFloat(hum),
                gas: parseFloat(gas),
                pm25: parseFloat(pm25) // Parse PM2.5
            };
        }).filter(f => f.timestamp && !isNaN(f.temperature) && !isNaN(f.humidity) && !isNaN(f.gas) && !isNaN(f.pm25));

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

app.get('/api/predict', async (req, res) => {
    try {
        const temp = parseFloat(req.query.temperature);
        const hum = parseFloat(req.query.humidity);
        const pm25 = parseFloat(req.query.pm25); // Get PM2.5 from query parameters

        if (isNaN(temp) || isNaN(hum)) {
            return res.status(400).json({
                success: false,
                error: 'Invalid parameters',
                details: 'Temperature and humidity must be numbers'
            });
        }

        // Include field4 for PM2.5 in the input CSV to Python
        const inputData = created_at,field1,field2,field4\n"${new Date().toISOString()}",${temp},${hum},${pm25};
        const inputFile = path.join(config.dataDir, 'predict_input.csv');
        fs.writeFileSync(inputFile, inputData);

        const result = await runPythonModel([
            '--predict', inputFile,
            '--data', path.join(config.dataDir, config.dataFile)
        ]);

        res.json({
            success: true,
            prediction: {
                gasLevel: result.gas_level,
                gasType: result.gas_type,
                airQuality: result.air_quality,
                pm25Quality: result.pm25_quality, // Include PM2.5 quality
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
            console.log(Server running on port ${PORT});
            console.log(Data updates every ${config.updateInterval/60000} minutes);
        });
    } catch (error) {
        console.error('Initialization failed:', error.message);
        process.exit(1);
    }
}

initialize();
