mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"kaisei92777@yahoo.co.jp\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml