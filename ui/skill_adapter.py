
def adapt_text(text, level="intermediate"):
    if level == "beginner":
        return f"🔰 Beginner: {text}\n💡 Tip: This is a basic debugging tip."
    elif level == "expert":
        return f"⚡ Expert: {text}\n🚀 Advanced optimization applied."
    else:
        return f"📊 Intermediate: {text}"
