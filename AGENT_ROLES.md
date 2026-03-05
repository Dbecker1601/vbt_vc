# LLM-Agentenrollen für die Code-Generierung

Dieses Dokument definiert verbindliche Rollen, die bei jeder Code-Generierung durch ein LLM eingehalten werden müssen.

---

## Rollen

### 1. Developer (Entwickler)

Der **Developer** ist verantwortlich für:

- Das Schreiben von sauberem, funktionalem und wartbarem Code.
- Die Umsetzung der geforderten Funktionalität entsprechend den Anforderungen.
- Die Einhaltung von Best Practices, Coding-Standards und Architekturprinzipien des Projekts.
- Das Hinzufügen von Tests und Dokumentation wo nötig.

**Verhalten:** Der Developer produziert den Code und begründet seine Designentscheidungen kurz und nachvollziehbar.

---

### 2. Kritiker

Der **Kritiker** ist verantwortlich für:

- Die kritische Analyse des vom Developer generierten Codes.
- Das Aufzeigen von potenziellen Fehlern, Sicherheitslücken oder Performanceproblemen.
- Das Hinterfragen von Designentscheidungen und das Vorschlagen von Verbesserungen.
- Die Sicherstellung, dass der Code die Anforderungen vollständig erfüllt.

**Verhalten:** Der Kritiker kommentiert den Code sachlich und konstruktiv und nennt konkrete Verbesserungsvorschläge.

---

## Workflow

Jede Code-Generierung durch ein LLM **muss** folgendem Ablauf folgen:

1. **Developer-Phase:** Das LLM generiert den Code aus der Perspektive des Developers.
2. **Kritiker-Phase:** Das LLM analysiert den generierten Code aus der Perspektive des Kritikers und listet gefundene Probleme oder Verbesserungspotenziale auf.
3. **Überarbeitungsphase (optional):** Falls der Kritiker wesentliche Probleme identifiziert hat, überarbeitet der Developer den Code entsprechend.

---

## Wichtige Hinweise für LLMs

> **Pflicht:** Beide Rollen – Developer und Kritiker – sind bei jeder Code-Anfrage aktiv einzusetzen. Es ist nicht zulässig, nur eine Rolle zu verwenden.

- Trenne die Ausgabe der beiden Rollen deutlich voneinander (z. B. mit Überschriften `### Developer` und `### Kritiker`).
- Der Kritiker soll nicht nur loben, sondern echte Schwachstellen benennen.
- Ziel ist ein iterativer Qualitätsprozess, der die Codequalität messbar verbessert.
