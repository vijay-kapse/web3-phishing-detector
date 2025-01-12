document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('message');
    const checkButton = document.getElementById('checkButton');
    const resultDiv = document.getElementById('result');

    checkButton.addEventListener('click', async () => {
        const message = messageInput.value.trim();
        if (!message) {
            resultDiv.textContent = 'Please enter a message to check.';
            resultDiv.classList.add('error');
            return;
        }

        resultDiv.textContent = 'Checking...';
        resultDiv.classList.remove('error');

        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message }),
            });

            if (!response.ok) {
                throw new Error('Server responded with an error');
            }

            const data = await response.json();
            resultDiv.textContent = `Prediction: ${data.prediction}, Probability: ${data.probability}`;
        } catch (error) {
            console.error('Error:', error);
            resultDiv.textContent = 'An error occurred while checking the message. Please try again.';
            resultDiv.classList.add('error');
        }
    });
});

