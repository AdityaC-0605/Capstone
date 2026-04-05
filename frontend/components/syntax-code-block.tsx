import React from "react";

type TokenType =
  | "key"
  | "string"
  | "number"
  | "punctuation"
  | "plain"
  | "space";

function tokenize(line: string) {
  const regex =
    /("(?:\\.|[^"])*")|(\btrue\b|\bfalse\b|\bnull\b|-?\d+(?:\.\d+)?)|([{}\[\]:,])|(\s+)|([^"\s{}\[\]:,]+)/g;
  const tokens: Array<{ value: string; type: TokenType }> = [];
  let match: RegExpExecArray | null;

  while ((match = regex.exec(line)) !== null) {
    if (match[1]) {
      tokens.push({ value: match[1], type: "string" });
    } else if (match[2]) {
      tokens.push({ value: match[2], type: "number" });
    } else if (match[3]) {
      tokens.push({ value: match[3], type: "punctuation" });
    } else if (match[4]) {
      tokens.push({ value: match[4], type: "space" });
    } else if (match[5]) {
      tokens.push({ value: match[5], type: "plain" });
    }
  }

  return tokens.map((token, index) => {
    if (token.type !== "string") return token;
    const nextVisible = tokens.slice(index + 1).find((candidate) => candidate.type !== "space");
    return {
      ...token,
      type: nextVisible?.value === ":" ? "key" : "string",
    };
  });
}

const classByType: Record<Exclude<TokenType, "space">, string> = {
  key: "syntax-key",
  string: "syntax-string",
  number: "syntax-number",
  punctuation: "syntax-punctuation",
  plain: "syntax-plain",
};

export function SyntaxCodeBlock({
  code,
  className,
}: {
  code: string;
  className?: string;
}) {
  const lines = code.split("\n");

  return (
    <pre className={className || "mt-4 overflow-x-auto text-xs leading-6"}>
      {lines.map((line, lineIndex) => (
        <React.Fragment key={`${line}-${lineIndex}`}>
          {tokenize(line).map((token, index) =>
            token.type === "space" ? (
              <span key={`${token.value}-${index}`}>{token.value}</span>
            ) : (
              <span
                key={`${token.value}-${index}`}
                className={classByType[token.type as Exclude<TokenType, "space">]}
              >
                {token.value}
              </span>
            ),
          )}
          {lineIndex < lines.length - 1 ? "\n" : null}
        </React.Fragment>
      ))}
    </pre>
  );
}
