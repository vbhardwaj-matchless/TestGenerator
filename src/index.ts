import express, { Request, Response } from "express";

const app = express();
app.use(express.json({ limit: "1mb" }));

export type UIChangeInput = {
  component?: string;
  action?: string;
  target?: string;
  severity?: "low" | "medium" | "high";
  notes?: string;
};

export type QualityCoachResponse = {
  normalized_input: {
    keywords: string[];
    summary: string;
    optional_fields: Partial<UIChangeInput>;
  };
  confidence: number;
  mode: "guided" | "clarify" | "investigate";
  risk: "low" | "medium" | "high";
  tests: Array<{
    tag: "happy_path" | "validation" | "recovery";
    title: string;
    steps: string[];
  }>;
};

type CoachRequest = {
  free_text?: string;
  optional_fields?: Partial<UIChangeInput>;
};

const keywordMap: Record<string, string> = {
  "btn": "button",
  "cta": "button",
  "signin": "login",
  "sign-in": "login",
  "signup": "register",
  "sign-up": "register",
  "auth": "authentication",
  "pw": "password",
  "pwd": "password",
  "checkout": "payment",
  "cart": "payment",
  "save": "submit",
  "remove": "delete",
  "erase": "delete",
  "ui": "interface"
};

const riskKeywords = {
  high: ["delete", "payment", "authentication", "password", "security", "pii"],
  medium: ["update", "edit", "submit", "upload", "download", "interface"],
  low: [] as string[]
};

const modeThresholds = {
  guided: 0.7,
  clarify: 0.4
};

const testTags: Array<"happy_path" | "validation" | "recovery"> = [
  "happy_path",
  "validation",
  "recovery"
];

function normalizeFreeText(freeText: string): { keywords: string[]; summary: string } {
  const cleaned = freeText
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!cleaned) {
    return { keywords: [], summary: "" };
  }

  const rawTokens = cleaned.split(" ");
  const normalizedTokens = rawTokens.map((token) => keywordMap[token] ?? token);
  const uniqueKeywords = Array.from(new Set(normalizedTokens)).filter(Boolean);
  const summary = normalizedTokens.slice(0, 12).join(" ");

  return { keywords: uniqueKeywords, summary };
}

function scoreConfidence(
  keywords: string[],
  freeText: string,
  optionalFields: Partial<UIChangeInput>
): number {
  let score = 0.35;
  if (freeText.trim().length > 0) {
    score += 0.15;
  }
  if (keywords.length >= 3) {
    score += 0.2;
  } else if (keywords.length > 0) {
    score += 0.1;
  }
  const optionalValues = Object.values(optionalFields).filter((value) => value !== undefined);
  if (optionalValues.length > 0) {
    score += 0.15;
  }
  if (freeText.trim().length > 80) {
    score += 0.1;
  }
  return Math.min(1, Number(score.toFixed(2)));
}

function selectMode(confidence: number): "guided" | "clarify" | "investigate" {
  if (confidence >= modeThresholds.guided) {
    return "guided";
  }
  if (confidence >= modeThresholds.clarify) {
    return "clarify";
  }
  return "investigate";
}

function classifyRisk(keywords: string[]): "low" | "medium" | "high" {
  if (keywords.some((keyword) => riskKeywords.high.includes(keyword))) {
    return "high";
  }
  if (keywords.some((keyword) => riskKeywords.medium.includes(keyword))) {
    return "medium";
  }
  return "low";
}

function deriveSubject(keywords: string[], optionalFields: Partial<UIChangeInput>): string {
  return (
    optionalFields.component ||
    keywords.find((keyword) => ["button", "form", "modal", "table", "banner"].includes(keyword)) ||
    "interface"
  );
}

function buildTests(
  subject: string,
  risk: "low" | "medium" | "high",
  mode: "guided" | "clarify" | "investigate"
): QualityCoachResponse["tests"] {
  const tests = testTags.map((tag) => {
    const baseSteps = [
      `Navigate to the ${subject} area.`,
      `Observe the ${subject} before applying changes.`,
      `Apply the intended change.`
    ];

    const riskStep =
      risk === "high"
        ? `Confirm that sensitive actions are gated with an extra confirmation.`
        : risk === "medium"
        ? `Verify that the update is saved and visible.`
        : `Confirm the update appears.`;

    const modeStep =
      mode === "investigate"
        ? "Record any ambiguous behavior for follow-up."
        : mode === "clarify"
        ? "Verify the updated copy aligns with the request."
        : "Validate the expected behavior end-to-end.";

    const steps = [...baseSteps, riskStep, modeStep];

    return {
      tag,
      title: `${tag.replace("_", " ")} for ${subject}`,
      steps
    };
  });

  return tests.slice(0, 3);
}

function validateResponse(response: QualityCoachResponse): QualityCoachResponse {
  const maxTests = 3;
  const allowedTags = new Set(testTags);

  const validTests = response.tests
    .slice(0, maxTests)
    .map((test) => ({
      ...test,
      steps: test.steps.filter((step) => step.trim().length > 0)
    }))
    .filter((test) => allowedTags.has(test.tag) && test.steps.length > 0);

  const normalizedInput = {
    keywords: response.normalized_input.keywords.filter(Boolean),
    summary: response.normalized_input.summary ?? "",
    optional_fields: response.normalized_input.optional_fields ?? {}
  };

  return {
    normalized_input: normalizedInput,
    confidence: Math.min(1, Math.max(0, response.confidence)),
    mode: response.mode,
    risk: response.risk,
    tests: validTests
  };
}

function buildResponse(payload: CoachRequest): QualityCoachResponse {
  const freeText = payload.free_text ?? "";
  const optionalFields = payload.optional_fields ?? {};
  const normalized = normalizeFreeText(freeText);
  const confidence = scoreConfidence(normalized.keywords, freeText, optionalFields);
  const mode = selectMode(confidence);
  const risk = classifyRisk(normalized.keywords);
  const subject = deriveSubject(normalized.keywords, optionalFields);

  const response: QualityCoachResponse = {
    normalized_input: {
      keywords: normalized.keywords,
      summary: normalized.summary,
      optional_fields: optionalFields
    },
    confidence,
    mode,
    risk,
    tests: buildTests(subject, risk, mode)
  };

  return validateResponse(response);
}

app.post("/coach-ui", (req: Request, res: Response) => {
  const payload = req.body as CoachRequest;
  const response = buildResponse(payload ?? {});
  res.json(response);
});

const port = Number(process.env.PORT ?? 3000);
app.listen(port, () => {
  console.log(`quality-assistant-coach listening on ${port}`);
});
