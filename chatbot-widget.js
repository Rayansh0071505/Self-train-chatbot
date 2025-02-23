(function() {
    const BACKEND_URL = 'http://localhost:8080';
    
    class ChatbotWidget {
        constructor(config) {
            this.widgetId = config.widgetId;
            this.theme = config.theme || 'light';
            this.createWidget();
        }

        createWidget() {
            // Create widget container
            const container = document.createElement('div');
            container.id = 'ai-chatbot-widget';
            container.style.cssText = `
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 380px;
                height: 600px;
                background: white;
                border-radius: 15px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                display: flex;
                flex-direction: column;
                overflow: hidden;
                z-index: 1000;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            `;

            // Create chat interface
            container.innerHTML = `
                <div style="padding: 20px; background: #1a73e8; border-bottom: 1px solid #e8eaed;">
                    <h3 style="margin: 0; color: white; font-size: 18px;">Chat Support</h3>
                </div>
                <div id="chat-messages" style="
                    flex: 1; 
                    overflow-y: auto; 
                    padding: 20px;
                    background: #f8f9fa;
                "></div>
                <div style="padding: 15px; border-top: 1px solid #e8eaed; background: white;">
                    <div style="display: flex; gap: 10px;">
                        <input type="text" id="chat-input" placeholder="Type your message..." 
                               style="
                                   flex: 1;
                                   padding: 12px;
                                   border: 1px solid #e8eaed;
                                   border-radius: 25px;
                                   font-size: 14px;
                                   outline: none;
                                   transition: border-color 0.3s;
                               ">
                        <button id="send-button" style="
                            background: #1a73e8;
                            border: none;
                            border-radius: 50%;
                            width: 40px;
                            height: 40px;
                            color: white;
                            cursor: pointer;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                        ">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
                            </svg>
                        </button>
                    </div>
                </div>
            `;

            document.body.appendChild(container);

            // Add event listeners
            const input = container.querySelector('#chat-input');
            const sendButton = container.querySelector('#send-button');

            const sendMessage = () => {
                const message = input.value.trim();
                if (message) {
                    this.sendMessage(message);
                    input.value = '';
                }
            };

            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });

            sendButton.addEventListener('click', sendMessage);
        }

        formatProductCard(product) {
            return `
                <div style="
                    border: 1px solid #e8eaed;
                    border-radius: 10px;
                    margin: 10px 0;
                    padding: 15px;
                    background: white;
                ">
                    ${product.image ? `
                        <img src="${product.image}" alt="${product.title}" style="
                            width: 100%;
                            height: 200px;
                            object-fit: cover;
                            border-radius: 8px;
                            margin-bottom: 10px;
                        ">
                    ` : ''}
                    <h4 style="margin: 0 0 8px 0; color: #1a73e8;">${product.title}</h4>
                    <div style="color: #1e8e3e; font-weight: bold; margin-bottom: 8px;">$${product.price}</div>
                    <p style="margin: 0; color: #5f6368; font-size: 14px;">
                        ${product.description.length > 100 
                            ? product.description.substring(0, 100) + '...' 
                            : product.description}
                    </p>
                </div>
            `;
        }

        async sendMessage(message) {
            const messagesContainer = document.querySelector('#chat-messages');
            
            // Add user message
            messagesContainer.innerHTML += `
                <div style="margin-bottom: 15px; text-align: right;">
                    <span style="
                        background: #1a73e8;
                        color: white;
                        padding: 10px 15px;
                        border-radius: 18px;
                        border-bottom-right-radius: 5px;
                        display: inline-block;
                        max-width: 80%;
                        word-wrap: break-word;
                        font-size: 14px;
                    ">${message}</span>
                </div>
            `;
            
            try {
                const response = await fetch(`${BACKEND_URL}/chat/${this.widgetId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: message,
                        user_id: this.widgetId
                    })
                });
                
                const data = await response.json();
                console.log("Response data:", data);  // Debug logging
                
                // Build the bot response HTML
                let responseHTML = `
                    <div style="margin-bottom: 15px;">
                        <span style="
                            background: #f1f3f4;
                            color: #202124;
                            padding: 10px 15px;
                            border-radius: 18px;
                            border-bottom-left-radius: 5px;
                            display: inline-block;
                            max-width: 80%;
                            word-wrap: break-word;
                            font-size: 14px;
                        ">${data.message}</span>
                    </div>
                `;
                
                if (data.type === "greeting") {
                    // Existing code for greeting
                    responseHTML += `<div style="margin-top: 10px;"><ul style="padding-left: 20px;">`;
                    data.results.forEach(option => {
                      if (option && option.category) {
                        responseHTML += `
                          <li style="margin-bottom: 5px; cursor: pointer;" 
                              onclick="document.querySelector('#chat-input').value='Category: ${option.category}'">
                              ${option.category}
                          </li>`;
                      }
                    });
                    responseHTML += `</ul></div>`;
                  
                  } else if (data.type === "vendor_selection") {
                    // NEW: Show vendor list
                    responseHTML += `<div style="margin-top: 10px;"><ul style="padding-left: 20px;">`;
                    data.results.forEach(option => {
                      if (option && option.vendor) {
                        responseHTML += `
                          <li style="margin-bottom: 5px; cursor: pointer;" 
                              onclick="document.querySelector('#chat-input').value='Vendor: ${option.vendor}'">
                              ${option.vendor}
                          </li>`;
                      }
                    });
                    responseHTML += `</ul></div>`;
                  
                  } else if (data.results && data.results.length > 0 && data.type === "product_query") {
                    // Existing code for product cards
                    responseHTML += `<div style="margin-top: 10px;">`;
                    data.results.forEach(product => {
                      responseHTML += this.formatProductCard(product);
                    });
                    responseHTML += `</div>`;
                  }
                  
                
                messagesContainer.innerHTML += responseHTML;
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            } catch (error) {
                console.error('Error sending message:', error);
                messagesContainer.innerHTML += `
                    <div style="margin-bottom: 15px; color: #d93025; font-size: 14px;">
                        Error: Could not send message. Please try again later.
                    </div>
                `;
            }
        }
        
        
    }

    // Initialize widget when the script loads
    window.addEventListener('load', () => {
        if (window.AIChatbot) {
            new ChatbotWidget(window.AIChatbot);
        }
    });
})();
