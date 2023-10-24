const express = require("express");
const { spawn } = require("child_process");

const app = express();

const executePython = async (script, args) => {
    // Get args
    const pyArgs = [
        '--start_scat', args[0].toString(),
        '--end_scat', args[1].toString(),
        '--date', args[2].toString(),
        '--time', args[3].toString(),
        '--model', args[4].toString()
      ];

    // Execute python script
    const py = spawn("python3", [script, ...pyArgs]);

    const result = await new Promise((resolve, reject) => {
        let output;

        // Get output from python script
        py.stdout.on("data", (data) => {
            output = JSON.parse(data);
        });

        // Handle any error occured
        py.stderr.on("data", (data) => {
            console.error(`[python] Error occured: ${data}`);
            // reject(new Error(`Error occured in ${script}`));
        });

        py.on("exit", (code) => {
            console.log(`Child process exited with code ${code}`);
            resolve(output);
        });
    });

    return result;
}

app.post("/", async (req, res) => {
    // Get inputs from frontend"s query strings
    const startScatNumber = req.query.start_scat;
    const endScatNumber = req.query.end_scat;
    const date = req.query.date;
    const timeOfDay = req.query.time;
    const predictionModel = req.query.model;

    // If inputs are missing, return error to frontend and show missing inputs to backend
    if (!startScatNumber || !endScatNumber || !date || !timeOfDay || !predictionModel) {
        const error = new Error("Missing parameters");
        error.statusCode = 400;
        console.log({
            startScatNumber,
            endScatNumber,
            date,
            timeOfDay,
            predictionModel
        });
        throw error;
    }

    // Else execute python script, using the inputs as arguments
    try {
        const result = await executePython("main.py", [startScatNumber, endScatNumber, date, timeOfDay, predictionModel]);
        res.json(result);
    } catch (error) {
        if (error.statusCode) {
            res.status(error.statusCode).send(error.message);
        }

        res.status(500).json({ error: error.message });
    }
});

app.listen(3001, () => {
    console.log("[server] Application started!")
});
