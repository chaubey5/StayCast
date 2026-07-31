function appState() {
    return {
        darkMode: false,
        mobileMenuOpen: false,
        loading: false,
        prediction: {
            age: null,
            sex: '',
            blood_pressure: null,
            cholesterol: null,
            heart_rate: null,
            diabetes: '',
        },
        contact: {
            name: '',
            email: '',
            message: '',
            status: '',
        },
        auth: {
            username: '',
            email: '',
            first_name: '',
            last_name: '',
            password: '',
            password2: '',
            status: '',
            error: '',
        },
        result: {},
        init() {
            this.loading = false;
            window.addEventListener('scroll', () => {
                const header = document.querySelector('.site-header');
                if (window.scrollY > 26) {
                    header.classList.add('sticky');
                } else {
                    header.classList.remove('sticky');
                }
            });
        },
        toggleTheme() {
            this.darkMode = !this.darkMode;
        },
        getCookie(name) {
            const cookies = document.cookie.split('; ');
            for (const cookie of cookies) {
                const [key, value] = cookie.split('=');
                if (key === name) {
                    return decodeURIComponent(value || '');
                }
            }
            return null;
        },
        async submitLogin() {
            this.loading = true;
            this.auth.error = '';
            const csrfToken = this.getCookie('csrftoken');
            try {
                const response = await fetch('/api/login/', {
                    method: 'POST',
                    credentials: 'same-origin',
                    headers: {
                        'Content-Type': 'application/json',
                        ...(csrfToken ? { 'X-CSRFToken': csrfToken } : {}),
                    },
                    body: JSON.stringify({
                        username: this.auth.username,
                        password: this.auth.password,
                    }),
                });
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.error || 'Login failed');
                }
                window.location.href = '/dashboard/';
            } catch (error) {
                this.auth.error = error.message;
            } finally {
                this.loading = false;
            }
        },
        async submitRegister() {
            this.loading = true;
            this.auth.error = '';
            const csrfToken = this.getCookie('csrftoken');
            try {
                const response = await fetch('/api/register/', {
                    method: 'POST',
                    credentials: 'same-origin',
                    headers: {
                        'Content-Type': 'application/json',
                        ...(csrfToken ? { 'X-CSRFToken': csrfToken } : {}),
                    },
                    body: JSON.stringify({
                        username: this.auth.username,
                        email: this.auth.email,
                        first_name: this.auth.first_name,
                        last_name: this.auth.last_name,
                        password: this.auth.password,
                        password2: this.auth.password2,
                    }),
                });
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.error || Object.values(data).flat().join(' '));
                }
                window.location.href = '/';
            } catch (error) {
                this.auth.error = error.message;
            } finally {
                this.loading = false;
            }
        },
        async fetchToken(username, password) {
            // Obtain JWT access/refresh tokens and store them in localStorage
            try {
                const resp = await fetch('/api/token/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password }),
                });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || 'Token request failed');
                localStorage.setItem('access_token', data.access);
                localStorage.setItem('refresh_token', data.refresh);
                return data;
            } catch (e) {
                console.error('Token fetch error', e);
                throw e;
            }
        },
        async submitPrediction() {
            this.loading = true;
            const payload = { ...this.prediction };
            const csrfToken = this.getCookie('csrftoken');
            try {
                const response = await fetch('/api/predict/', {
                    method: 'POST',
                    credentials: 'same-origin',
                    headers: {
                        'Content-Type': 'application/json',
                        ...(csrfToken ? { 'X-CSRFToken': csrfToken } : {}),
                    },
                    body: JSON.stringify(payload),
                });
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.error || 'Prediction failed');
                }
                this.result = {
                    title: data.prediction || 'Prediction complete',
                    prediction: data.prediction,
                    probability: data.probability,
                    explanation: data.explanation,
                };
            } catch (error) {
                this.result = {
                    title: 'Error',
                    prediction: '',
                    probability: 0,
                    explanation: error.message,
                };
            } finally {
                this.loading = false;
            }
        },
        async submitContact() {
            this.loading = true;
            const csrfToken = this.getCookie('csrftoken');
            try {
                const response = await fetch('/api/contact/', {
                    method: 'POST',
                    credentials: 'same-origin',
                    headers: {
                        'Content-Type': 'application/json',
                        ...(csrfToken ? { 'X-CSRFToken': csrfToken } : {}),
                    },
                    body: JSON.stringify(this.contact),
                });
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.error || 'Unable to send message');
                }
                this.contact.status = data.message || 'Message sent successfully!';
                this.contact.name = '';
                this.contact.email = '';
                this.contact.message = '';
            } catch (error) {
                this.contact.status = error.message;
            } finally {
                this.loading = false;
            }
        },
        resetPrediction() {
            this.prediction = {
                age: null,
                sex: '',
                blood_pressure: null,
                cholesterol: null,
                heart_rate: null,
                diabetes: '',
            };
            this.result = {};
        },
        renderHistoryChart(labels = ['High', 'Medium', 'Low'], values = [3, 2, 1]) {
            const canvas = document.querySelector('#historyChart');
            if (!canvas) return;

            new Chart(canvas, {
                type: 'doughnut',
                data: {
                    labels,
                    datasets: [{
                        data: values,
                        backgroundColor: ['#ff6f8f', '#f9b34a', '#3de7a1'],
                        borderWidth: 0,
                    }],
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: true, position: 'bottom', labels: { color: '#abb7ce' } },
                        tooltip: { callbacks: { label: (ctx) => `${ctx.label}: ${ctx.formattedValue}` } },
                    },
                },
            });
        },
    };
}
