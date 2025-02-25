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

            // Create chat interface with improved styling
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
                        ${product.description.length > 100 ? 
                            product.description.substring(0, 100) + '...' : 
                            product.description}
                    </p>
                </div>
            `;
        }
        
        // Extract brands from a message string using various patterns
        extractBrands(message) {
            // Try to find numbered list format first (1. Brand)
            const numberedListPattern = /\d+\.\s*([a-zA-Z0-9 ]+)(?=(?:\s+\d+\.|$))/g;
            let matches = [...message.matchAll(numberedListPattern)];
            
            if (matches.length > 0) {
                return matches.map(match => match[1].trim());
            }
            
            // If that fails, look for brands following a colon
            const colonPattern = /(?:brands|brand).*?:\s*(.*)/i;
            const colonMatch = message.match(colonPattern);
            
            if (colonMatch) {
                // Split by commas or "and"
                return colonMatch[1].split(/,|\sand\s/).map(brand => brand.trim());
            }
            
            // As a fallback, look for any capitalized words that might be brands
            const brandNames = ["adidas", "nike", "puma", "reebok", "asics", "converse", 
                                "dr martens", "new balance", "vans", "under armour"];
            
            const foundBrands = [];
            for (const brand of brandNames) {
                if (message.toLowerCase().includes(brand.toLowerCase())) {
                    foundBrands.push(brand);
                }
            }
            
            return foundBrands;
        }
        
        // Create brand selection UI
        createBrandSelection(brands) {
            // Create a container for the brand options
            const container = document.createElement('div');
            container.className = 'brand-selection-container';
            container.style.cssText = `
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                margin-top: 15px;
                margin-bottom: 15px;
            `;
            
            // Add brand option cards
            brands.forEach((brand, index) => {
                const card = document.createElement('div');
                card.className = 'brand-option';
                card.style.cssText = `
                    background: white;
                    border: 1px solid #e0e0e0;
                    border-radius: 12px;
                    padding: 15px;
                    text-align: center;
                    cursor: pointer;
                    transition: all 0.2s;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                `;
                
                // Add number
                const numberElement = document.createElement('div');
                numberElement.style.cssText = `
                    font-weight: bold;
                    color: #1a73e8;
                    margin-bottom: 5px;
                `;
                numberElement.textContent = (index + 1).toString();
                
                // Add brand name
                const nameElement = document.createElement('div');
                nameElement.style.cssText = `
                    color: #202124;
                `;
                nameElement.textContent = brand;
                
                // Add to card
                card.appendChild(numberElement);
                card.appendChild(nameElement);
                
                // Add hover effects
                card.addEventListener('mouseover', () => {
                    card.style.borderColor = '#1a73e8';
                    card.style.boxShadow = '0 4px 8px rgba(0,0,0,0.1)';
                });
                
                card.addEventListener('mouseout', () => {
                    card.style.borderColor = '#e0e0e0';
                    card.style.boxShadow = '0 2px 4px rgba(0,0,0,0.05)';
                });
                
                // Add click handler
                card.addEventListener('click', () => {
                    const input = document.querySelector('#chat-input');
                    const sendButton = document.querySelector('#send-button');
                    
                    if (input && sendButton) {
                        input.value = brand;
                        sendButton.click();
                    }
                });
                
                // Add to container
                container.appendChild(card);
            });
            
            return container;
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
            
            messagesContainer.scrollTop = messagesContainer.scrollHeight;

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
                
                // Build bot response
                const botMessageContainer = document.createElement('div');
                botMessageContainer.style.marginBottom = '15px';
                
                // Add text message
                const textSpan = document.createElement('span');
                textSpan.style.cssText = `
                    background: #f1f3f4;
                    color: #202124;
                    padding: 10px 15px;
                    border-radius: 18px;
                    border-bottom-left-radius: 5px;
                    display: inline-block;
                    max-width: 80%;
                    word-wrap: break-word;
                    font-size: 14px;
                `;
                textSpan.textContent = data.message;
                botMessageContainer.appendChild(textSpan);
                messagesContainer.appendChild(botMessageContainer);

                // Check for brand selection in the message
                const messageContainsBrandSelection = 
                    (data.type === "brand_selection") || 
                    data.message.includes("select one") ||
                    data.message.match(/\d+\.\s+[a-zA-Z]/) !== null;
                
                if (messageContainsBrandSelection) {
                    // Extract brands from the message
                    const brands = this.extractBrands(data.message);
                    
                    if (brands.length > 0) {
                        console.log("Found brands:", brands);
                        // Create and add brand selection UI
                        const brandSelection = this.createBrandSelection(brands);
                        messagesContainer.appendChild(brandSelection);
                    }
                }
                // Handle product results
                else if (data.results && data.results.length > 0) {
                    const resultsContainer = document.createElement('div');
                    resultsContainer.style.marginTop = '10px';
                    
                    if (data.type === "product_query" || data.type === "show_more" || data.type === "deep_search") {
                        // Render product cards
                        data.results.forEach(product => {
                            resultsContainer.innerHTML += this.formatProductCard(product);
                        });
                    }
                    
                    messagesContainer.appendChild(resultsContainer);
                }
                
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            } catch (error) {
                console.error('Error sending message:', error);
                const errorDiv = document.createElement('div');
                errorDiv.style.cssText = "margin-bottom: 15px; color: #d93025; font-size: 14px;";
                errorDiv.textContent = "Error: Could not send message. Please try again later.";
                messagesContainer.appendChild(errorDiv);
            }
        }
    }

    // Initialize widget on load
    window.addEventListener('load', () => {
        if (window.AIChatbot) {
            new ChatbotWidget(window.AIChatbot);
        }
    });
})();