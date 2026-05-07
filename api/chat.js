module.exports = async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    const { messages, problem } = req.body;
    if (!messages || !Array.isArray(messages)) {
        return res.status(400).json({ error: 'Invalid request' });
    }

    const systemPrompt = `You are a coding interview assistant helping users on TopAlgorithms.com.
${problem ? `The user is currently working on: "${problem}".` : ''}
Help them understand algorithms, data structures, time/space complexity, and problem-solving approaches.
Be concise, educational, and encouraging. Use code examples when helpful.
Do not just give away the full solution — guide them to think through it.`;

    try {
        const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: 'llama-3.3-70b-versatile',
                messages: [
                    { role: 'system', content: systemPrompt },
                    ...messages.slice(-12),
                ],
                max_tokens: 1024,
                temperature: 0.7,
            }),
        });

        if (!response.ok) {
            const err = await response.text();
            console.error('Groq error:', err);
            return res.status(502).json({ error: 'AI service error — try again' });
        }

        const data = await response.json();
        const reply = data.choices?.[0]?.message?.content || '';
        res.status(200).json({ reply });

    } catch (e) {
        console.error('Handler error:', e);
        res.status(500).json({ error: 'Server error — try again' });
    }
};
