import streamlit as st


def main():
    st.title("MCP Journlist")

    with st.sidebar:
        st.header("Settings")
        source_type = st.selectbox(
            "Data Sources",
            options=["both", "news", "reddit"],
            format_func=lambda x: (
                f"🕸️ { x.capitalize() }" if x == "news" else f"🔍 { x.capitalize() }"
            ),
        )


if __name__ == "__main__":
    main()
