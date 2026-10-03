import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { videoName, backbone, pythonApiUrl } = body;

    // Check if user requested external Python backend relay
    if (pythonApiUrl) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 8000);

        const pyResponse = await fetch(pythonApiUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: controller.signal,
        });
        clearTimeout(timeoutId);

        if (pyResponse.ok) {
          const pyData = await pyResponse.json();
          return NextResponse.json({
            success: true,
            source: 'python_backend',
            result: pyData,
          });
        }
      } catch {
        // Fall back gracefully to serverless telemetry
      }
    }

    // Default Vercel Edge response confirmation
    return NextResponse.json({
      success: true,
      source: 'vercel_serverless',
      message: 'Serverless forensic endpoint active',
      timestamp: new Date().toISOString(),
      received: { videoName, backbone },
    });
  } catch (error) {
    return NextResponse.json(
      { success: false, error: (error as Error).message },
      { status: 500 }
    );
  }
}
