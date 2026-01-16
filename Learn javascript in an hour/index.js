

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('todo-form');
    const input = document.getElementById('todo-input');
    const list = document.getElementById('todo-list');

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        const task = input.value.trim();
        if (task) {
            const li = document.createElement('li');
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.addEventListener('change', function() {
                if (checkbox.checked) {
                    li.style.textDecoration = 'line-through';
                } else {
                    li.style.textDecoration = '';
                }
            });
            li.appendChild(checkbox);
            li.appendChild(document.createTextNode(' ' + task));
            list.appendChild(li);
            input.value = '';
        }
    });
});