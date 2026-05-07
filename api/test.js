module.exports = async function handler(req, res) {
    const keySet = !!process.env.GROQ_API_KEY;
    const keyPrefix = process.env.GROQ_API_KEY ? process.env.GROQ_API_KEY.substring(0, 8) + '...' : 'NOT SET';

    // Actually try calling Groq
    let groqStatus = 'not tested';
    try {
        const r = await fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: 'llama-3.3-70b-versatile',
                messages: [{ role: 'user', content: 'say hi' }],
                max_tokens: 10,
            }),
        });
        const body = await r.text();
        groqStatus = `HTTP ${r.status}: ${body.substring(0, 200)}`;
    } catch (e) {
        groqStatus = `fetch error: ${e.message}`;
    }

    res.status(200).json({ keySet, keyPrefix, groqStatus });
};
